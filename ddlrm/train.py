import torch
import argparse
import logging
from torch import nn
from torch.utils._pytree import tree_map
from dataset import make_random_data_and_loader
from model import DLRM


class TracedTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, tensor: torch.Tensor):
        result = super(TracedTensor, cls)._make_wrapper_subclass(
            cls, tensor.shape, dtype=tensor.dtype,
            requires_grad=tensor.requires_grad, device=tensor.device)
        return result

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(e) -> torch.Tensor:
            return e.tensor if isinstance(e, TracedTensor) else e

        print(f"> S dispatch {func}")
        result = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))

        if func is torch.ops.aten.detach.default:
            return TracedTensor(result)
        return result

    @staticmethod
    def wrap_model(model: nn.Module) -> None:
        for name, param in model.named_parameters():
            module_name = '.'.join(name.split('.')[:-1])
            param_name = name.split('.')[-1]
            setattr(model.get_submodule(module_name), param_name,
                    torch.nn.Parameter(TracedTensor(param)))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Deep Learning Recommendation Model (DLRM)")

    # Dataset args
    parser_ds = parser.add_argument_group("Dataset Args")
    parser_ds.add_argument(
        "--data-size", type=int, default=100)
    parser_ds.add_argument(
        "--num-batches", type=int, default=0)
    parser_ds.add_argument(
        "--mini-batch-size", type=int, default=1)
    parser_ds.add_argument(
        "--num-indices-per-lookup", type=int, default=10)
    parser_ds.add_argument(
        "--num-indices-per-lookup-fixed", type=bool, default=False)
    parser_ds.add_argument(
        "--num-workers", type=int, default=0)

    # Model args
    parser_md = parser.add_argument_group("Model Args")
    parser_md.add_argument(
        "--arch-sparse-feature-size", type=int, default=2)
    parser_md.add_argument(
        "--arch-interaction-op", type=str, choices=["dot", "cat"], default="cat")

    # Model/Dataset shared args
    parser_shared = parser.add_argument_group("Model/Dataset Shared Args")
    parser_shared.add_argument(
        "--arch-embedding-size", type=int, nargs='+', default=[4, 3, 2])
    parser_shared.add_argument(
        "--arch-mlp-bot", type=int, nargs='+', default=[4, 3, 2])
    parser_shared.add_argument(
        "--arch-mlp-top", type=int, nargs='+', default=[8, 4, 2, 1])

    # Trainig args
    parser_tr = parser.add_argument_group("Training Args")
    parser_tr.add_argument(
        "--nepochs", type=int, default=10)
    parser.add_argument(
        "--learning-rate", type=float, default=0.01)

    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.DEBUG)
    args = parse_args()

    # Data
    _, train_loader, _, _ = make_random_data_and_loader(args)

    # Model
    dlrm_model = DLRM(
        m_spa=args.arch_sparse_feature_size,
        ln_emb=args.arch_embedding_size,
        ln_bot=args.arch_mlp_bot,
        ln_top=args.arch_mlp_top,
        arch_interaction_op=args.arch_interaction_op
    )
    logging.info(dlrm_model)
    TracedTensor.wrap_model(dlrm_model)

    compiled = torch.compile(dlrm_model, fullgraph=True, dynamic=True)
    batch = next(iter(train_loader))
    X, lS_o, lS_i, T = batch[0], batch[1], batch[2], batch[3]
    compiled(X, lS_o, lS_i).sum().backward()

    logging.info("Done")

if __name__ == "__main__":
    main()
