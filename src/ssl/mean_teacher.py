import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanTeacher:
    """Maintains an exponential-moving-average copy of a student model.

    Usage:
        mt = MeanTeacher(student, alpha=0.999)
        ...inside training loop...
        student_logits = student(x)
        with torch.no_grad():
            teacher_logits = mt.teacher(x_perturbed)
        loss = sup_loss + consistency_loss(student_logits, teacher_logits)
        loss.backward(); optimizer.step()
        mt.update(student)
    """

    def __init__(self, student: nn.Module, alpha: float = 0.999):
        self.alpha = alpha
        self.teacher = copy.deepcopy(student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    @torch.no_grad()
    def update(self, student: nn.Module) -> None:
        for tp, sp in zip(self.teacher.parameters(), student.parameters()):
            tp.data.mul_(self.alpha).add_(sp.detach().data, alpha=1.0 - self.alpha)


def consistency_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """MSE between sigmoid outputs of student and teacher."""
    return F.mse_loss(torch.sigmoid(student_logits), torch.sigmoid(teacher_logits))
