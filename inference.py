import os 
import dotenv
dotenv.load_dotenv()

import fire
import vllm
import json
from tqdm import tqdm
import pandas as pd

def main(
    dataset_path: str = './data/prompt.csv',           # 데이터셋 파일의 경로
    output_file: str = './results/prompt.jsonl',       # 결과물 저장 경로
    base_model_id='google/gemma-3-1b-it', # 베이스 모델 ID (예: google/gemma-3-1b-it)
    batch_size = 30,    # 배치 사이즈
    max_tokens = 64,    # 생성할 최대 토큰 수
):
    # 데이터셋 로드 및 저장 경로 전처리
    dataset = pd.read_csv(dataset_path) # 데이터셋 로드
    os.makedirs(os.path.dirname(output_file), exist_ok=True) # 결과물 저장 경로 생성 (폴더 없으면 생성)
    if os.path.exists(output_file): # 이미 결과가 있으면 종료
        print(f"Output file {output_file} already exists. Exiting...")
        return
    
    # 생성 파라미터 설정
    sampling_params = vllm.SamplingParams(
        temperature=0.1,
        top_p=0.75,
        max_tokens=max_tokens,
    )
    
    # 모델 선언
    llm = vllm.LLM(model=base_model_id, task="generate", enforce_eager=True)
    
    # 입력 데이터 형태 구성
    input_list = []
    question_only_list = []
    for i in tqdm(range(len(dataset)), desc=f"Processing", total=len(dataset)):
        query = dataset['prompt'][i]        
        prompt = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": f"{dataset['prompt'][i]}",
            },
        ]
            
        input_list.append(prompt)
        question_only_list.append(query)

    result_dict = []
    for i in tqdm(range(0, len(dataset), batch_size), desc=f"Generating responses", total=len(dataset)//batch_size):
        batch_prompt = input_list[i:i+batch_size] # 입력 배치 구성
        
        # 생성 시킴
        output = llm.chat(
            batch_prompt, 
            sampling_params=sampling_params,
            # use_tqdm=False, # 주석 해제 시 progress bar on
        )
        
        # 결과물 중간 저장
        for j in range(i, min(i+batch_size, len(dataset))):
            index = j - i
            result_dict.append({
                'query': question_only_list[j], # 데이터셋에 있는 질문
                'answer': output[index].outputs[0].text, # 모델의 생성물
                'prompt': input_list[j], # 실제로 들어간 프롬프트 형태
            })
        
        # 중간 중간에 결과물을 저장
        with open(output_file, 'w') as f:
            json.dump(result_dict, f, indent=4)

    with open(output_file, 'w') as f:
        json.dump(result_dict, f, indent=4)
    print(f"Results saved to {output_file}")
    return 

if __name__ == '__main__':
    fire.Fire(main)
