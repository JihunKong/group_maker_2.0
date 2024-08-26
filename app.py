import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from openai import OpenAI

# OpenAI API 키 설정
try:
    openai_api_key = st.secrets["openai"]["api_key"]
    client = OpenAI(api_key=openai_api_key)
except KeyError:
    st.error("OpenAI API 키가 올바르게 설정되지 않았습니다. Streamlit의 secrets에서 [openai] 섹션 아래에 api_key를 설정해주세요.")
    st.stop()

def load_data(uploaded_file):
    df = pd.read_excel(uploaded_file)
    df['성적'] = df['성적'].apply(lambda x: float(x.split(':')[1]) if isinstance(x, str) and ':' in x else np.nan)
    return df

def create_balanced_groups(df, group_size=4):
    df_sorted = df.sort_values('성적', ascending=False)
    num_students = len(df)
    num_groups = num_students // group_size
    if num_students % group_size != 0:
        num_groups += 1
    
    groups = [[] for _ in range(num_groups)]
    
    # 성적 기반 초기 분배
    for i, (_, student) in enumerate(df_sorted.iterrows()):
        group_index = i % num_groups
        groups[group_index].append(student)
    
    # 학습 스타일, MBTI, 관심사를 고려한 추가 조정
    for _ in range(2):  # 두 번의 추가 조정 라운드
        for i in range(num_groups):
            for j in range(i+1, num_groups):
                if improve_group_balance(groups[i], groups[j]):
                    break
    
    return groups

def improve_group_balance(group1, group2):
    # 두 그룹 간의 균형을 개선하는 로직
    # 학습 스타일, MBTI, 관심사의 다양성을 고려
    styles1 = set(s['학습스타일'] for s in group1)
    styles2 = set(s['학습스타일'] for s in group2)
    mbti1 = set(s['MBTI'] for s in group1)
    mbti2 = set(s['MBTI'] for s in group2)
    
    if len(styles1) < len(styles2) or len(mbti1) < len(mbti2):
        for i, student1 in enumerate(group1):
            for j, student2 in enumerate(group2):
                if (student1['학습스타일'] not in styles2 or student2['학습스타일'] not in styles1 or
                    student1['MBTI'] not in mbti2 or student2['MBTI'] not in mbti1):
                    group1[i], group2[j] = group2[j], group1[i]
                    return True
    return False

def format_groups(groups):
    formatted_groups = []
    for i, group in enumerate(groups, 1):
        group_df = pd.DataFrame(group)
        group_df = group_df.sort_values('성적', ascending=False)
        group_df['역할'] = ['모둠장'] + [''] * (len(group_df) - 1)
        group_df = group_df[['이름', '학급', '번호', '성적', '학습스타일', 'MBTI', '관심사', '협업능력', '디지털리터러시', '역할']]
        formatted_groups.append(f"모둠 {i}:\n{group_df.to_string(index=False)}\n")
    return "\n".join(formatted_groups)

def get_gpt_instruction(groups):
    prompt = f"""
다음은 학생들의 모둠 편성 결과입니다. 이를 바탕으로 각 모둠별 특징과 조언을 제공해주세요.
편성 결과:
{format_groups(groups)}

다음 사항들을 고려해주세요:
1. 각 모둠의 평균 성적과 성적 분포
2. 학습 스타일의 다양성
3. MBTI 유형의 조합
4. 관심사의 다양성
5. 협업 능력과 디지털 리터러시 수준
6. 모둠장(성적 최상위자)의 특성

각 모둠에 대해 '모둠 1:', '모둠 2:' 등의 형식으로 시작하여 간략한 분석과 조언을 제공해주세요.
모둠 활동 시 고려해야 할 점과 각 모둠의 강점, 보완해야 할 점 등을 언급해주세요.
"""
    
    with st.spinner('GPT 분석 및 조언을 생성 중입니다...'):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 교육 전문가이며 학생들의 모둠 활동을 돕는 조언자입니다."},
                {"role": "user", "content": prompt}
            ]
        )
    
    return response.choices[0].message.content

def main():
    st.title("개선된 국어 모둠 편성 도우미")
    
    uploaded_file = st.file_uploader("엑셀 파일을 업로드하세요", type="xlsx")
    
    if uploaded_file is not None:
        with st.spinner('파일을 처리 중입니다...'):
            df = load_data(uploaded_file)
        
        st.write("업로드된 데이터:")
        st.write(df)
        
        group_size = st.number_input("모둠 크기를 입력하세요", min_value=2, max_value=6, value=4)
        
        if st.button("모둠 편성하기"):
            with st.spinner('모둠을 편성 중입니다...'):
                groups = create_balanced_groups(df, group_size)
            
            st.write("모둠 편성 결과:")
            for i, group in enumerate(groups, 1):
                st.write(f"모둠 {i}:")
                group_df = pd.DataFrame(group)
                group_df = group_df.sort_values('성적', ascending=False)
                group_df['역할'] = ['모둠장'] + [''] * (len(group_df) - 1)
                st.write(group_df[['이름', '학급', '번호', '성적', '학습스타일', 'MBTI', '관심사', '협업능력', '디지털리터러시', '역할']])
            
            gpt_analysis = get_gpt_instruction(groups)
            st.write("GPT 분석 및 조언:")
            st.write(gpt_analysis)
            
            # 엑셀 파일로 저장
            with st.spinner('엑셀 파일을 생성 중입니다...'):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    # 전체 학생 명단 시트
                    df.to_excel(writer, sheet_name='전체 학생 명단', index=False)
                    
                    # 각 모둠별 시트
                    for i, group in enumerate(groups, 1):
                        group_df = pd.DataFrame(group)
                        group_df = group_df.sort_values('성적', ascending=False)
                        group_df['역할'] = ['모둠장'] + [''] * (len(group_df) - 1)
                        sheet_name = f'모둠 {i}'
                        group_df[['이름', '학급', '번호', '성적', '학습스타일', 'MBTI', '관심사', '협업능력', '디지털리터러시', '역할']].to_excel(writer, sheet_name=sheet_name, index=False)
                        
                        # GPT 조언 추가
                        worksheet = writer.sheets[sheet_name]
                        worksheet.write(len(group_df) + 2, 0, 'GPT 분석 및 조언:')
                        group_advice = gpt_analysis.split(f"모둠 {i}:")[1].split(f"모둠 {i+1}:")[0].strip()
                        worksheet.write(len(group_df) + 3, 0, group_advice)
                
                output.seek(0)
            
            st.download_button(
                label="엑셀 파일 다운로드",
                data=output,
                file_name="모둠_편성_결과.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    main()
