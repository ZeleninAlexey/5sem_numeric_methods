"""Выбор подлабораторной первой лабораторной работы."""
import argparse
import importlib
import sys
from numerics import validate_eps


def main():
    parser = argparse.ArgumentParser(description="Лабораторная 1, примеры 10")
    parser.add_argument("--task", choices=["1", "2", "3", "4", "5", "all"], default="1")
    parser.add_argument("--example", action="store_true", help="использовать пример 10")
    parser.add_argument("--eps", type=float, default=1e-6, help="точность итерационных методов")
    args = parser.parse_args()
    try:
        validate_eps(args.eps)
        tasks = range(1, 6) if args.task == "all" else [int(args.task)]
        for task in tasks:
            print(f"\nПодлабораторная 1.{task}")
            module = importlib.import_module(f"lab1_{task}")
            if task == 1:
                # Первая подлабораторная сохраняет прежний интерфейс ввода.
                previous = sys.argv[:]
                sys.argv = [previous[0]] + (["--example"] if args.example else [])
                try:
                    status = module.main()
                finally:
                    sys.argv = previous
            else:
                status = module.main(example=args.example, eps=args.eps)
            if status:
                return status
    except (ValueError, OverflowError) as error:
        print(f"Ошибка: {error}")
        return 1
    except (EOFError, KeyboardInterrupt):
        print("\nВвод прерван.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
