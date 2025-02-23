import pandas as pd
import fire
import os
import glob
import tqdm

def process_data(data_path: str, output_path: str = None):
    if output_path is None:
        output_path = data_path
    
    file_list = glob.glob(os.path.join(data_path, '**/*.csv'), recursive=True)
    for file in tqdm.tqdm(file_list, "Processing files"):
        data = pd.read_csv(file)
        if data.shape[1] > 2:
            data = data.iloc[:, :-3]
        data.to_csv(file, index=False)

if __name__ == "__main__":
    fire.Fire(process_data)