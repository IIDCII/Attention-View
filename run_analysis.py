from inference_analysis import Qwen2VLExplorer


def main():
    model_path = "Qwen/Qwen2-VL-2B-Instruct"
    Explorer = Qwen2VLExplorer(model_path=model_path)

    Explorer.print_model_structure()


if __name__ == "__main__":
    main()
