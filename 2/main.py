import argparse
import lab2_1
import lab2_2
from numerics import read_eps, validate_eps


def main():
    parser = argparse.ArgumentParser(description="Лабораторная работа 2, вариант 10")
    parser.add_argument("--task", choices=["1", "2", "all"], default="1")
    parser.add_argument("--example", action="store_true", help="вариант 10 с точностью 1e-6 без ручного ввода")
    parser.add_argument("--eps", type=float, help="точность вычислений")
    args = parser.parse_args()
    try:
        eps = args.eps
        if eps is None:
            eps = 1e-6 if args.example else read_eps()
        validate_eps(eps)
        tasks = [1, 2] if args.task == "all" else [int(args.task)]
        for task in tasks:
            print(f"\nПодлабораторная 2.{task}, eps = {eps:g}")
            module = lab2_1 if task == 1 else lab2_2
            module.main(eps)
    except (ValueError, OverflowError) as error:
        print(f"Ошибка: {error}")
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\nВвод прерван.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
