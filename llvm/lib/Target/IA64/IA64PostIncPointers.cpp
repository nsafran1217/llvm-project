//===-- IA64PostIncPointers.cpp - Give accesses pointers of their own -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A load or store has no offset, only an immediate it may add to its address
// register afterwards. Instruction selection folds the step of a pointer into
// one access through it, but a loop unrolled to read a[i], a[i + 1], ... and
// to write a[i] back has one pointer for them all, and computes the other
// addresses with adds in each iteration: in daxpy unrolled four times there
// are five adds for twelve accesses.
//
// IA64PostIncPointers gives each access in a single-block loop at a constant
// offset from a pointer that steps by a constant a pointer of its own, which
// starts at that offset in the preheader and is stepped by the access itself.
// The adds that formed the offsets go. So does the one that stepped the
// pointer, the first access at no offset stepping it instead, or where there
// is none and nothing else uses the pointer, with the pointer itself. Each
// new pointer holds a register for the whole loop, so a loop gains only a
// few, and none where no add goes. The accesses keep their memory operands,
// from which the pipeliner relates the new pointers to each other.
//
// In any other block, accesses at constant offsets from a pointer are chained
// where that leaves the critical path of the block no longer: the first keeps
// its add, and each steps the address on to the next, so that the rounds of a
// cipher reading its key schedule off a context pointer take one add rather
// than one for each word.
//
//===----------------------------------------------------------------------===//

#include "IA64.h"
#include "IA64InstrInfo.h"
#include "IA64Subtarget.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetSchedule.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "ia64-postinc-pointers"

STATISTIC(NumPointers, "Number of accesses given a pointer of their own");
STATISTIC(NumAdds, "Number of address adds removed from loops");
STATISTIC(NumChained, "Number of address adds replaced by a chain");
STATISTIC(NumUnfolded, "Number of post-increments split from their access");

static cl::opt<bool>
    EnablePostIncPointers("ia64-enable-postinc-pointers", cl::init(true),
                          cl::Hidden,
                          cl::desc("Give each access in a loop a pointer of "
                                   "its own that it steps itself"));

static cl::opt<unsigned>
    MaxPointers("ia64-postinc-max-pointers", cl::init(16), cl::Hidden,
                cl::desc("The most pointers to add to one loop"));

static cl::opt<unsigned> ChainGap(
    "ia64-postinc-chain-gap", cl::init(2), cl::Hidden,
    cl::desc("The fewest instructions from one access of a chain of accesses "
             "in a block to the next, 0 to chain none"));

static cl::opt<bool> UnfoldLiveBases(
    "ia64-unfold-live-post-inc", cl::init(true), cl::Hidden,
    cl::desc("Split the step from an access whose address register is used "
             "after it"));

static cl::opt<unsigned>
    MaxChains("ia64-postinc-max-chains", cl::init(2), cl::Hidden,
              cl::desc("The most chains of accesses in a block off a pointer"));

namespace {
class IA64PostIncPointers : public MachineFunctionPass {
public:
  static char ID;

  IA64PostIncPointers() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "IA64 post-increment pointers";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setIsSSA();
  }

private:
  const IA64InstrInfo *TII = nullptr;
  TargetSchedModel SchedModel;
  MachineRegisterInfo *MRI = nullptr;

  bool runOnLoop(MachineBasicBlock &MBB, MachineBasicBlock &Preheader);
  bool runOnBlock(MachineBasicBlock &MBB);
  bool unfoldLiveBases(MachineFunction &MF);
  bool isLiveOut(Register Reg, const MachineBasicBlock &MBB) const;
  bool isAccessThrough(const MachineInstr &MI, Register Base) const;
  bool isOffset(const MachineInstr &MI, Register Base,
                const MachineBasicBlock &MBB) const;
  void step(MachineInstr &Access, Register Base, Register NewBase,
            int64_t Step);
};
} // end anonymous namespace

char IA64PostIncPointers::ID = 0;

INITIALIZE_PASS(IA64PostIncPointers, DEBUG_TYPE, "IA64 post-increment pointers",
                false, false)

// The form of a load or store that adds an immediate to its address
// register, or 0 for one that has none.
static unsigned getPostIncOpcode(unsigned Opcode) {
  switch (Opcode) {
  default:
    return 0;
#define ACCESS(LOAD, STORE)                                                    \
  case IA64::LOAD:                                                             \
    return IA64::LOAD##_i;                                                     \
  case IA64::STORE:                                                            \
    return IA64::STORE##_i;
    ACCESS(LD1, ST1)
    ACCESS(LD2, ST2)
    ACCESS(LD4, ST4)
    ACCESS(LD8, ST8)
    ACCESS(LDFS, STFS)
    ACCESS(LDFD, STFD)
    ACCESS(LDFE, STFE)
#undef ACCESS
  }
}

static bool isUnpredicated(const MachineInstr &MI) {
  const MachineOperand &QP = MI.getOperand(MI.getNumExplicitOperands() - 1);
  return QP.isReg() && QP.getReg() == IA64::P0;
}

// The amount a load or store adds to its address register after the access,
// which follows the address of a load and the value of a store.
static std::optional<int64_t> getPostIncrement(const MachineInstr &MI,
                                               Register Base) {
  if (!MI.mayLoadOrStore() || !MI.hasOneMemOperand())
    return std::nullopt;
  unsigned Idx = MI.getNumExplicitDefs();
  const MachineOperand &BaseOp = MI.getOperand(Idx);
  if (!BaseOp.isReg() || BaseOp.getReg() != Base || !BaseOp.isTied())
    return std::nullopt;
  const MachineOperand &Inc = MI.getOperand(Idx + (MI.mayStore() ? 2 : 1));
  if (!Inc.isImm())
    return std::nullopt;
  return Inc.getImm();
}

// The plain form of a load or store that adds an immediate or a register to
// its address register, the opcode that does the add, or {0, 0}.
static std::pair<unsigned, unsigned> getUnfoldedOpcodes(unsigned Opcode) {
  switch (Opcode) {
  default:
    return {0, 0};
#define LOAD(LOAD)                                                             \
  case IA64::LOAD##_i:                                                         \
    return {IA64::LOAD, IA64::ADDS};                                           \
  case IA64::LOAD##_r:                                                         \
    return {IA64::LOAD, IA64::ADD};
#define STORE(STORE)                                                           \
  case IA64::STORE##_i:                                                        \
    return {IA64::STORE, IA64::ADDS};
    LOAD(LD1)
    LOAD(LD2)
    LOAD(LD4)
    LOAD(LD8)
    LOAD(LDFS)
    LOAD(LDFD)
    LOAD(LDFE)
    STORE(ST1)
    STORE(ST2)
    STORE(ST4)
    STORE(ST8)
    STORE(STFS)
    STORE(STFD)
    STORE(STFE)
#undef LOAD
#undef STORE
  }
}

// Whether MI is a plain load or store that uses Base as its address and for
// nothing else.
bool IA64PostIncPointers::isAccessThrough(const MachineInstr &MI,
                                          Register Base) const {
  if (!getPostIncOpcode(MI.getOpcode()) || !MI.hasOneMemOperand() ||
      MI.hasOrderedMemoryRef() || !isUnpredicated(MI))
    return false;
  unsigned Idx = MI.getNumExplicitDefs();
  if (MI.getOperand(Idx).getReg() != Base)
    return false;
  for (unsigned I = 0, E = MI.getNumOperands(); I != E; ++I)
    if (I != Idx && MI.getOperand(I).isReg() &&
        MI.getOperand(I).getReg() == Base)
      return false;
  return true;
}

// Whether MI adds a constant to Base to form addresses that only accesses in
// MBB use.
bool IA64PostIncPointers::isOffset(const MachineInstr &MI, Register Base,
                                   const MachineBasicBlock &MBB) const {
  if (MI.getOpcode() != IA64::ADDS || !isUnpredicated(MI) ||
      MI.getOperand(2).getReg() != Base ||
      !isInt<14>(MI.getOperand(1).getImm()))
    return false;
  Register Addr = MI.getOperand(0).getReg();
  return all_of(MRI->use_nodbg_instructions(Addr), [&](MachineInstr &U) {
    return U.getParent() == &MBB && isAccessThrough(U, Addr);
  });
}

// Replace Access with its form that adds Step to Base to give NewBase.
void IA64PostIncPointers::step(MachineInstr &Access, Register Base,
                               Register NewBase, int64_t Step) {
  unsigned Idx = Access.getNumExplicitDefs();
  MachineInstrBuilder MIB =
      BuildMI(*Access.getParent(), Access, Access.getDebugLoc(),
              TII->get(getPostIncOpcode(Access.getOpcode())));
  if (Access.mayLoad())
    MIB.add(Access.getOperand(0)).addDef(NewBase).addReg(Base);
  else
    MIB.addDef(NewBase).addReg(Base).add(Access.getOperand(Idx + 1));
  MIB.addImm(Step)
      .add(Access.getOperand(Access.getNumExplicitOperands() - 2))
      .add(Access.getOperand(Access.getNumExplicitOperands() - 1))
      .cloneMemRefs(Access)
      .setMIFlags(Access.getFlags());
  Access.eraseFromParent();
}

bool IA64PostIncPointers::runOnLoop(MachineBasicBlock &MBB,
                                    MachineBasicBlock &Preheader) {
  bool Changed = false;
  unsigned Added = 0;
  SmallVector<MachineInstr *, 8> Phis;
  for (MachineInstr &Phi : MBB.phis())
    if (MRI->getRegClass(Phi.getOperand(0).getReg()) == &IA64::GRRegClass)
      Phis.push_back(&Phi);

  for (MachineInstr *Phi : Phis) {
    Register Ptr = Phi->getOperand(0).getReg(), Init, Next;
    for (unsigned I = 1, E = Phi->getNumOperands(); I != E; I += 2)
      (Phi->getOperand(I + 1).getMBB() == &MBB ? Next : Init) =
          Phi->getOperand(I).getReg();
    MachineInstr *Step = MRI->getVRegDef(Next);
    if (!Step || Step->getParent() != &MBB)
      continue;
    int64_t Stride;
    bool StepIsAdd = Step->getOpcode() == IA64::ADDS;
    if (StepIsAdd) {
      if (!isUnpredicated(*Step) || Step->getOperand(2).getReg() != Ptr)
        continue;
      Stride = Step->getOperand(1).getImm();
    } else if (std::optional<int64_t> Inc = getPostIncrement(*Step, Ptr);
               Inc && Step->getOperand(Step->getNumExplicitDefs() - 1)
                              .getReg() == Next) {
      // The pointer is the address the access steps, not what a load reads
      // through it: in n = n->next, ld8 next = [n], 28 steps n by 28 for an
      // access at that offset, but the loop goes on from the value loaded.
      Stride = *Inc;
    } else {
      continue;
    }
    if (!isInt<9>(Stride))
      continue;

    SmallVector<MachineInstr *, 4> Direct, Offsets;
    bool Others = false;
    for (MachineInstr &U : MRI->use_nodbg_instructions(Ptr)) {
      if (&U == Step)
        continue;
      if (U.getParent() == &MBB && isAccessThrough(U, Ptr))
        Direct.push_back(&U);
      else if (U.getParent() == &MBB && isOffset(U, Ptr, MBB))
        Offsets.push_back(&U);
      else
        Others = true;
    }
    // The add that steps the pointer goes if the first access at no offset
    // can step it instead, which it can if nothing in the loop reads the
    // stepped pointer before that access. With no access at no offset, the
    // add goes only if the pointer goes too, nothing else using either.
    MachineInstr *First = nullptr;
    if (StepIsAdd && !Direct.empty())
      for (MachineInstr &MI : MBB) {
        if (is_contained(Direct, &MI)) {
          First = &MI;
          break;
        }
        if (!MI.isPHI() && MI.readsRegister(Next, /*TRI=*/nullptr))
          break;
      }
    bool StepGoes = First || (StepIsAdd && Direct.empty() && !Others &&
                              MRI->hasOneNonDBGUse(Next));
    unsigned Needed = 0;
    for (MachineInstr *Offset : Offsets)
      Needed +=
          std::distance(MRI->use_nodbg_begin(Offset->getOperand(0).getReg()),
                        MRI->use_nodbg_end());
    if (First)
      Needed += Direct.size() - 1;
    if ((Offsets.empty() && !StepGoes) || Added + Needed > MaxPointers)
      continue;
    Added += Needed;

    // Accesses to the same address start from the same register.
    SmallDenseMap<int64_t, Register, 4> Starts;
    Starts[0] = Init;
    auto NewPointer = [&](MachineInstr &Access, int64_t Offset) {
      Register &Start = Starts[Offset];
      if (!Start) {
        Start = MRI->createVirtualRegister(&IA64::GRRegClass);
        BuildMI(Preheader, Preheader.getFirstTerminator(), DebugLoc(),
                TII->get(IA64::ADDS), Start)
            .addImm(Offset)
            .addReg(Init)
            .addReg(IA64::P0);
      }
      Register Base = MRI->createVirtualRegister(&IA64::GRRegClass);
      Register NewBase = MRI->createVirtualRegister(&IA64::GRRegClass);
      BuildMI(MBB, MBB.begin(), DebugLoc(), TII->get(IA64::PHI), Base)
          .addReg(Start)
          .addMBB(&Preheader)
          .addReg(NewBase)
          .addMBB(&MBB);
      step(Access, Base, NewBase, Stride);
      ++NumPointers;
    };

    for (MachineInstr *Offset : Offsets) {
      Register Addr = Offset->getOperand(0).getReg();
      int64_t Imm = Offset->getOperand(1).getImm();
      SmallVector<MachineInstr *, 4> Accesses(
          make_pointer_range(MRI->use_nodbg_instructions(Addr)));
      for (MachineInstr *Access : Accesses)
        NewPointer(*Access, Imm);
      MRI->markUsesInDebugValueAsUndef(Addr);
      Offset->eraseFromParent();
      ++NumAdds;
    }
    if (StepGoes) {
      Step->eraseFromParent();
      ++NumAdds;
      if (First) {
        step(*First, Ptr, Next, Stride);
        for (MachineInstr *Access : Direct)
          if (Access != First)
            NewPointer(*Access, 0);
      } else {
        MRI->markUsesInDebugValueAsUndef(Next);
        MRI->markUsesInDebugValueAsUndef(Ptr);
        Phi->eraseFromParent();
      }
    }
    Changed = true;
  }
  return Changed;
}

// Chain accesses in MBB at constant offsets from a pointer. Each link holds
// up the next until it has stepped the address, a cycle in which it would
// otherwise have gone: bh's force calculation waits on loads that a chain
// spreads out, so a link goes in only if the later access can wait that long
// in the critical path of the block, counted through registers. A second
// chain takes what the first cannot. A call ends the chains: an address held
// across it would cost a register that the adds do not.
bool IA64PostIncPointers::runOnBlock(MachineBasicBlock &MBB) {
  struct Link {
    MachineInstr *Access;
    int64_t Offset;
  };
  struct Chain {
    SmallVector<Link, 8> Links;
    unsigned Last;
  };
  SmallDenseMap<Register, SmallVector<Chain, 2>, 8> Open;
  bool Changed = false;

  // The cycle each instruction can go in at the earliest, and the cycles from
  // it to the end of the block on the longest path through it.
  DenseMap<const MachineInstr *, unsigned> Depth, Height;
  unsigned Critical = 0;
  auto ForEachEdge = [&](MachineInstr &MI, auto Fn) {
    for (auto [I, MO] : enumerate(MI.operands())) {
      if (!MO.isReg() || !MO.isUse() || !MO.getReg().isVirtual())
        continue;
      MachineInstr *Def = MRI->getVRegDef(MO.getReg());
      if (!Def || Def->getParent() != &MBB || Def->isPHI())
        continue;
      unsigned DefIdx = Def->findRegisterDefOperandIdx(MO.getReg(), nullptr);
      Fn(*Def, SchedModel.computeOperandLatency(Def, DefIdx, &MI, I));
    }
  };
  for (MachineInstr &MI : MBB) {
    if (MI.isDebugInstr() || MI.isPHI())
      continue;
    unsigned &D = Depth[&MI];
    ForEachEdge(MI, [&](MachineInstr &Def, unsigned Latency) {
      D = std::max(D, Depth.lookup(&Def) + Latency);
    });
  }
  for (MachineInstr &MI : reverse(MBB)) {
    if (MI.isDebugInstr() || MI.isPHI())
      continue;
    unsigned H = Height.lookup(&MI);
    ForEachEdge(MI, [&](MachineInstr &Def, unsigned Latency) {
      unsigned &DH = Height[&Def];
      DH = std::max(DH, H + Latency);
    });
    Critical = std::max(Critical, Depth.lookup(&MI) + H);
  }

  auto Finish = [&](Chain &C) {
    if (C.Links.size() < 2)
      return;
    Changed = true;
    unsigned Idx = C.Links[0].Access->getNumExplicitDefs();
    Register Cur = C.Links[0].Access->getOperand(Idx).getReg();
    for (unsigned I = 0, E = C.Links.size(); I != E; ++I) {
      MachineInstr *Access = C.Links[I].Access;
      unsigned BaseIdx = Access->getNumExplicitDefs();
      Register Addr = Access->getOperand(BaseIdx).getReg();
      if (I + 1 != E) {
        Register Next = MRI->createVirtualRegister(&IA64::GRRegClass);
        step(*Access, Cur, Next, C.Links[I + 1].Offset - C.Links[I].Offset);
        Cur = Next;
      } else {
        Access->getOperand(BaseIdx).setReg(Cur);
        Access->getOperand(BaseIdx).setIsKill(false);
      }
      if (I != 0) {
        MachineInstr *Add = MRI->getVRegDef(Addr);
        MRI->markUsesInDebugValueAsUndef(Addr);
        Add->eraseFromParent();
        ++NumChained;
      }
    }
  };
  auto FinishAll = [&] {
    for (auto &Entry : Open)
      for (Chain &C : Entry.second)
        Finish(C);
    Open.clear();
  };

  unsigned Pos = 0;
  for (auto I = MBB.begin(), E = MBB.end(); I != E; ++I) {
    MachineInstr &MI = *I;
    if (MI.isDebugInstr())
      continue;
    ++Pos;
    if (MI.isCall()) {
      FinishAll();
      continue;
    }
    if (!getPostIncOpcode(MI.getOpcode()))
      continue;
    Register Addr = MI.getOperand(MI.getNumExplicitDefs()).getReg();
    if (!Addr.isVirtual() || !MRI->hasOneNonDBGUse(Addr))
      continue;
    MachineInstr *Add = MRI->getVRegDef(Addr);
    if (!Add || Add->getOpcode() != IA64::ADDS || !Add->getOperand(1).isImm() ||
        !Add->getOperand(2).isReg())
      continue;
    Register Base = Add->getOperand(2).getReg();
    if (!Base.isVirtual() || !isOffset(*Add, Base, MBB) ||
        !isAccessThrough(MI, Addr))
      continue;
    int64_t Offset = Add->getOperand(1).getImm();
    SmallVector<Chain, 2> &Chains = Open[Base];
    Chain *Into = nullptr;
    for (Chain &C : Chains)
      if (Pos - C.Last >= ChainGap &&
          isInt<9>(Offset - C.Links.back().Offset) &&
          Depth.lookup(C.Links.back().Access) + 1 + Height.lookup(&MI) <=
              Critical) {
        Into = &C;
        // MI goes no earlier than this now, nor can the next link.
        unsigned &D = Depth[&MI];
        D = std::max(D, Depth.lookup(C.Links.back().Access) + 1);
        break;
      }
    if (!Into) {
      if (Chains.size() >= MaxChains)
        continue;
      Into = &Chains.emplace_back();
    }
    Into->Links.push_back({&MI, Offset});
    Into->Last = Pos;
  }
  FinishAll();
  return Changed;
}

// An access that steps its address register writes the new address over the
// old one, so where the old address is used after the access as well,
// register allocation copies it first, to another register for the access to
// step. The copy takes a group of its own ahead of the access: in the
// dispatch of CPython's interpreter loop, a mov ahead of the load of the
// opcode, which steps the instruction pointer for the load of the argument.
// Split the step from such an access into an add beside it instead, which
// leaves the access where it was and gives the add the same register to read.
// Whether Reg, which is in SSA form, is used on some path out of MBB before
// the path reaches its definition again. A PHI reads it at the end of the
// block it comes in from.
bool IA64PostIncPointers::isLiveOut(Register Reg,
                                    const MachineBasicBlock &MBB) const {
  const MachineBasicBlock *DefMBB = MRI->getVRegDef(Reg)->getParent();
  SmallPtrSet<const MachineBasicBlock *, 8> UseBlocks, PHIPreds;
  for (const MachineOperand &MO : MRI->use_nodbg_operands(Reg)) {
    const MachineInstr &U = *MO.getParent();
    if (U.isPHI())
      PHIPreds.insert(U.getOperand(MO.getOperandNo() + 1).getMBB());
    else
      UseBlocks.insert(U.getParent());
  }
  if (PHIPreds.contains(&MBB))
    return true;
  SmallVector<const MachineBasicBlock *, 8> Worklist(MBB.successors());
  SmallPtrSet<const MachineBasicBlock *, 8> Visited;
  while (!Worklist.empty()) {
    const MachineBasicBlock *B = Worklist.pop_back_val();
    if (B == DefMBB || !Visited.insert(B).second)
      continue;
    if (UseBlocks.contains(B) || PHIPreds.contains(B))
      return true;
    append_range(Worklist, B->successors());
  }
  return false;
}

bool IA64PostIncPointers::unfoldLiveBases(MachineFunction &MF) {
  SmallVector<MachineInstr *, 8> Unfold;
  for (MachineBasicBlock &MBB : MF) {
    // Post-increments in this block whose address register has no use after
    // them yet, by address register.
    SmallDenseMap<Register, SmallVector<MachineInstr *, 2>, 8> Pending;
    for (MachineInstr &MI : MBB) {
      if (MI.isDebugInstr())
        continue;
      for (const MachineOperand &MO : MI.uses()) {
        if (!MO.isReg() || !MO.getReg().isVirtual())
          continue;
        auto It = Pending.find(MO.getReg());
        if (It == Pending.end())
          continue;
        append_range(Unfold, It->second);
        Pending.erase(It);
      }
      if (!getUnfoldedOpcodes(MI.getOpcode()).first || !isUnpredicated(MI))
        continue;
      Register Base = MI.getOperand(MI.getNumExplicitDefs()).getReg();
      if (Base.isVirtual())
        Pending[Base].push_back(&MI);
    }
    for (auto &[Base, Accesses] : Pending)
      if (isLiveOut(Base, MBB))
        append_range(Unfold, Accesses);
  }

  for (MachineInstr *MI : Unfold) {
    auto [Plain, AddOpc] = getUnfoldedOpcodes(MI->getOpcode());
    unsigned Defs = MI->getNumExplicitDefs();
    Register NewBase = MI->getOperand(Defs - 1).getReg();
    Register Base = MI->getOperand(Defs).getReg();
    unsigned IncIdx = Defs + (MI->mayStore() ? 2 : 1);
    MachineBasicBlock &MBB = *MI->getParent();
    MachineInstrBuilder Access =
        BuildMI(MBB, *MI, MI->getDebugLoc(), TII->get(Plain));
    for (unsigned I = 0, E = MI->getNumExplicitOperands(); I != E; ++I) {
      if (I == Defs - 1 || I == IncIdx)
        continue;
      MachineOperand MO = MI->getOperand(I);
      if (MO.isReg() && MO.isUse())
        MO.setIsKill(false);
      Access.add(MO);
    }
    Access.cloneMemRefs(*MI);
    BuildMI(MBB, *MI, MI->getDebugLoc(), TII->get(AddOpc), NewBase)
        .add(MI->getOperand(IncIdx))
        .addReg(Base)
        .addReg(IA64::P0);
    MI->eraseFromParent();
    ++NumUnfolded;
  }
  return !Unfold.empty();
}

bool IA64PostIncPointers::runOnMachineFunction(MachineFunction &MF) {
  if (!EnablePostIncPointers || skipFunction(MF.getFunction()))
    return false;
  TII = MF.getSubtarget<IA64Subtarget>().getInstrInfo();
  MRI = &MF.getRegInfo();
  SchedModel.init(&MF.getSubtarget());
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    if (!MBB.isSuccessor(&MBB)) {
      if (ChainGap)
        Changed |= runOnBlock(MBB);
      continue;
    }
    if (MBB.pred_size() != 2)
      continue;
    MachineBasicBlock *Preheader = *find_if(
        MBB.predecessors(), [&](MachineBasicBlock *P) { return P != &MBB; });
    if (Preheader == &MBB)
      continue;
    Changed |= runOnLoop(MBB, *Preheader);
  }
  if (UnfoldLiveBases)
    Changed |= unfoldLiveBases(MF);
  return Changed;
}

FunctionPass *llvm::createIA64PostIncPointersPass() {
  return new IA64PostIncPointers();
}
