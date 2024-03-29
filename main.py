from transformers import (AutoTokenizer,
                          BartForConditionalGeneration)
import streamlit as st

def run(text: str) -> str:

    tokenizer = AutoTokenizer.from_pretrained("fnlp/bart-base-chinese")
    # 加载训练好的模型
    model = BartForConditionalGeneration.from_pretrained("results/best")
    model = model.to("cuda")

    inputs = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    # 生成
    outputs = model.generate(input_ids, 
                             attention_mask=attention_mask, 
                             max_length=256)
    # 将token转换为文字
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    output_str = [s.replace(" ","") for s in output_str]
    return output_str

def main():
    txt = st.text_area('原始文本', height=260)
    txt = run(txt)
    if st.button('run'):
        st.write('摘要文本:', txt[0])
    else:
        st.write('摘要文本:  ')



if __name__ == '__main__':
    main()