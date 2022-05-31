import torch


def build_loss_fn(config):

    scale = config.scale
    reduce_op = torch.sum

    def score_matching_loss(scores, x, x_pert):

        target = (x_pert - x) / scale ** 2
        loss = scores + target
        loss = loss.reshape(x.shape[0], -1)
        loss = torch.sum(loss ** 2, dim=-1)

        return reduce_op(loss)

    return score_matching_loss
