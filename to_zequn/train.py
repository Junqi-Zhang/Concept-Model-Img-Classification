import sys
import torch
from tqdm import tqdm


def run_epoch(desc, model, dataloader, classes_idx, train=False, metric_prefix=""):
    # train pipeline
    if train:
        model.train()
    else:
        model.eval()

    metric_dict = dict()

    step = 0
    with tqdm(
        total=len(dataloader),
        desc=desc,
        postfix=dict,
        mininterval=1,
        file=sys.stdout,
        dynamic_ncols=True
    ) as pbar:
        for data, targets in dataloader:
            # data process
            data = data.to(device)
            targets = targets.to(device)

            # forward pass
            if train:
                returned_dict = model(data)
                loss, loss_cls_per_img, loss_img_per_cls, loss_sparsity, loss_diversity = compute_loss(
                    returned_dict, targets, train=True
                )

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    returned_dict = model(data)
                    loss, loss_cls_per_img, loss_img_per_cls, loss_sparsity, loss_diversity = compute_loss(
                        returned_dict, targets, train=False
                    )

            # display the metrics
            with torch.no_grad():

                acc = (torch.argmax(returned_dict["outputs"].data,
                                    1) == targets).sum() / targets.size(0)

                mask = torch.zeros_like(returned_dict["outputs"].data)
                mask[:, classes_idx] = 1
                acc_subset = (torch.argmax(returned_dict["outputs"].data * mask,
                                           1) == targets).sum() / targets.size(0)

                if returned_dict.get("attention_weights", None) is not None:
                    attended_concepts_count = torch.sum(
                        (returned_dict.get("attention_weights").data - 1e-7) > 0,
                        dim=1
                    ).type(torch.float)
                    s10 = torch.quantile(attended_concepts_count, 0.10).item()
                    s50 = torch.quantile(attended_concepts_count, 0.50).item()
                    s90 = torch.quantile(attended_concepts_count, 0.90).item()
                else:
                    s10 = -1
                    s50 = -1
                    s90 = -1

            def update_metric_dict(key, value, average=True):
                if average:
                    metric_dict[metric_prefix + key] = (
                        metric_dict.get(
                            metric_prefix + key, 0
                        ) * step + value
                    ) / (step + 1)
                else:
                    metric_dict[metric_prefix + key] = value

            update_metric_dict("acc", acc.item())
            update_metric_dict("acc_subset", acc_subset.item())
            update_metric_dict("loss", loss.item())
            update_metric_dict("loss_cpi", loss_cls_per_img.item())
            update_metric_dict("loss_ipc", loss_img_per_cls.item())
            update_metric_dict("loss_dvs", loss_diversity.item())
            update_metric_dict("loss_sps", loss_sparsity.item())
            update_metric_dict(
                "loss_sps_w", config.loss_sparsity_weight, average=False
            )
            update_metric_dict("s10", s10)
            update_metric_dict("s50", s50)
            update_metric_dict("s90", s90)

            pbar.set_postfix(metric_dict)
            pbar.update(1)

            step += 1
    return metric_dict
