import torch


class FaceFormerLoss:
    def __init__(self) -> None:
        # self.mse = torch.nn.MSELoss()
        self.loss = VocaLoss()

    def __call__(self, pred, gt):
        gt = gt.squeeze(0)
        pred = pred.squeeze(0)
        # drop last frame if the number of frames is odd
        if gt.shape[0] % 2 != 0:
            gt = gt[:-1]
            pred = pred[:-1]

        return self.loss(pred, gt)

        # rec_loss = self.mse(pred, gt)

        # return {"loss": rec_loss, "rec_loss": rec_loss}


class VocaLoss:
    def __init__(self, k_rec: float = 1.0, k_vel: float = 10.0):
        self.k_rec = k_rec
        self.k_vel = k_vel

    def reconstruction_loss(self, pred, gt):
        return torch.mean(torch.sum((pred - gt) ** 2, axis=2))

    def velocity_loss(self, pred, gt):
        n_consecutive_frames = 2
        pred = pred.view(-1, n_consecutive_frames, self.n_verts, 3)
        gt = gt.view(-1, n_consecutive_frames, self.n_verts, 3)

        v_pred = pred[:, 1] - pred[:, 0]
        v_gt = gt[:, 1] - gt[:, 0]

        return torch.mean(torch.sum((v_pred - v_gt) ** 2, axis=2))

    def __call__(self, pred, gt):
        bs = pred.shape[0]
        gt = gt.view(bs, -1, 3)
        pred = pred.view(bs, -1, 3)
        self.n_verts = pred.shape[1]

        rec_loss = self.reconstruction_loss(pred, gt)
        vel_loss = self.velocity_loss(pred, gt)

        return {
            "loss": rec_loss * self.k_rec + vel_loss * self.k_vel,
            "rec_loss": rec_loss,
            "vel_loss": vel_loss,
        }
