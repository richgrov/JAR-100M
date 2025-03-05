import pandas as pd

for i in range(27):
    df = pd.read_parquet(f"dataset/train-{i:05}-of-00027.parquet")
    print(f"Writing {i:03}")
    with open(f"dataset/{i:03}.txt", "w") as file:
        for source in df["content"]:
            file.write(source)
        
