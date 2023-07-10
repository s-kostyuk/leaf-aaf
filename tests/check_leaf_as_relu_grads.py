import torch


def main():
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(mode=True)

    u = torch.randn(1, 99)
    p1 = 1.0
    p2 = 0.0
    p3 = 1e5
    p4 = 0.0

    print(u)

    grad_u = (
            (p1 * torch.sigmoid(p3 * u))
            +
            (p1 * u + p2)
            * torch.sigmoid(p3 * u)
            * torch.sigmoid(-p3 * u)
            * p3)

    grad_p1 = (u * torch.sigmoid(p3 * u))
    grad_p2 = torch.sigmoid(p3 * u)
    grad_p3 = (
        (p1 * u + p2)
        * torch.sigmoid(p3 * u)
        * torch.sigmoid(-p3 * u)
        * u
    )

    print(grad_u)
    print(grad_p1)
    print(grad_p2)
    print(grad_p3)


if __name__ == "__main__":
    main()
