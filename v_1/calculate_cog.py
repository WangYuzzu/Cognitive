import openai
import json
from typing import Optional


class StudentInfoExtractor:
    """
    提取学生信息的类，基于 LLM 调用解析自然语言描述。
    """
    def __init__(self, api_key: str):
        openai.api_key = api_key

    @staticmethod
    def _extract_json(text: str) -> str:
        """
        从文本中提取 JSON 字符串
        """
        text = text.strip()
        if text.startswith('{') and text.endswith('}'):
            return text
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return text[start:end + 1]
        raise ValueError(f"No valid JSON found in response, 对应text: {text}")

    def _extract_json_from_text(self, content: str) -> Optional[dict]:
        """
        从 LLM 返回内容中提取 JSON 数据
        """
        try:
            data = json.loads(self._extract_json(content))
            required_keys = ["age", "region", "is_key_school"]
            extracted_data = {key: data.get(key) for key in required_keys}

            if None in extracted_data.values():
                print(f"Missing required keys in LLM response: {extracted_data}")
                return None
            return extracted_data
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {str(e)}")
            return None

    def call_llm_api(self, prompt: str, max_retries: int = 3) -> Optional[dict]:
        """
        同步调用 LLM API，提取学生信息
        """
        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    api_base="https://api.gptapi.us/v1/chat/completions",
                    messages=[
                        {"role": "system", "content": "You are an expert in extracting structured information."},
                        {"role": "user", "content": f"""
                        根据以下描述提取学生信息，并以 JSON 格式输出：
                        - "age": 学生年龄，整数。
                        - "region": 学生所在的省份，字符串（如果是市区或者县，则要给出对应省份）。
                        - "is_key_school": 是否来自重点学校，布尔值（true 或 false）。

                        示例描述：
                        "这是一位15岁的学生，来自广东，所在学校是重点学校。"
                        示例输出：
                        {{
                            "age": 15,
                            "region": "广东",
                            "is_key_school": true
                        }}

                        请根据以下描述提取信息："{prompt}"
                        """}
                    ],
                    temperature=0
                )
                content = response['choices'][0]['message']['content']
                print(f"LLM 返回内容：\n{content}")
                return self._extract_json_from_text(content)
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    continue
                else:
                    raise e
        return None


class CognitionCalculator:
    """
    计算学生认知值的类
    """
    @staticmethod
    def calculate_cognition(age: int, region: str, is_key_school: bool) -> float:
        """
        根据学生信息计算认知值
        """
        age_mapping = [
            (3, 6, 1),
            (7, 9, 3),
            (10, 12, 5),
            (13, 15, 7),
            (16, 18, 9),
        ]
        region_mapping = {
            "北京": 1, "上海": 1, "江苏": 1, "天津": 1,
            "福建": 0.9, "河南": 0.9, "河北": 0.9, "山西": 0.9,
            "江西": 0.9, "湖北": 0.9, "湖南": 0.9, "广东": 0.9,
            "安徽": 0.9, "山东": 0.9, "重庆": 0.9,
            "甘肃": 0.8, "青海": 0.8, "内蒙古": 0.8, "黑龙江": 0.8,
            "吉林": 0.8, "辽宁": 0.8, "宁夏": 0.8, "海南": 0.8,
            "陕西": 0.8, "四川": 0.8,
            "云南": 0.7, "广西": 0.7, "贵州": 0.7, "新疆": 0.7,
            "西藏": 0.7,
        }
        base_cognition = next((value for start, end, value in age_mapping if start <= age <= end), 0)
        region_multiplier = region_mapping.get(region, 1.0)
        cognition = base_cognition * region_multiplier
        if age >= 10 and is_key_school:
            cognition += 1
        return cognition


class CognitionEvaluator:
    """
    集成 StudentInfoExtractor 和 CognitionCalculator 的类
    """
    def __init__(self, api_key: str):
        self.extractor = StudentInfoExtractor(api_key)
        self.calculator = CognitionCalculator()

    def evaluate(self, description: str) -> Optional[float]:
        """
        综合评估学生认知值
        """
        student_info = self.extractor.call_llm_api(description)
        if student_info:
            print("提取的学生信息：", student_info)
            return self.calculator.calculate_cognition(
                age=student_info.get("age"),
                region=student_info.get("region"),
                is_key_school=student_info.get("is_key_school")
            )
        else:
            print("未能成功提取学生信息")
            return None


def test():
    api_key = "sk-bCY6IwFu0nbiaRTF528e01387bFf4c3eB2E224B6201129A5"
    evaluator = CognitionEvaluator(api_key)
    description = "我今年根号144岁，在漯河上小学，虽然我想去重点学校，但我的学校很普通"
    cognition_value = evaluator.evaluate(description)
    if cognition_value is not None:
        print("计算得到的认知值：", cognition_value)


if __name__ == "__main__":
    test()