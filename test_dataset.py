

from datasets import load_dataset



def main():

    # 下载数据
    dataset = load_dataset("stanfordnlp/imdb", split="train")

    # 保存到本地 Arrow 格式
    dataset.save_to_disk("./imdb_train")

    # 加载本地数据
    from datasets import load_from_disk
    dataset_local = load_from_disk("./imdb_train")


if __name__ == "__main__":
    main()
