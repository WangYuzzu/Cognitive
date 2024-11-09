# stu_info_construction.py
import json
import asyncio
import aiohttp
from typing import Dict, List, Any
import time
from hashlib import md5
from datetime import datetime


class StudentInfoGenerator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.gptapi.us/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.student_infos = set()  # 用于存储学生信息的哈希值

    def _hash_student_info(self, info: Dict) -> str:
        """生成学生信息的哈希值，用于去重"""
        # 选择关键字段进行哈希
        key_fields = {
            "age": info["age"],
            "grade": info["grade"],
            "academic_performance": info["academic_performance"],
            "interests": sorted(info["interests"]),
            "family_background": {
                "parents_education": info["family_background"]["parents_education"],
                "education_support": info["family_background"]["education_support"]
            },
            "learning_traits": info["learning_traits"]
        }
        return md5(json.dumps(key_fields, sort_keys=True).encode()).hexdigest()

    def _is_duplicate(self, info: Dict) -> bool:
        """检查是否是重复的学生信息"""
        info_hash = self._hash_student_info(info)
        if info_hash in self.student_infos:
            return True
        self.student_infos.add(info_hash)
        return False

    @staticmethod
    def _extract_json_from_text(text: str) -> str:
        """从文本中提取JSON字符串"""
        # 去除可能的前后缀文本
        text = text.strip()

        # 如果文本已经是一个完整的JSON，直接返回
        if text.startswith('{') and text.endswith('}'):
            return text

        # 寻找第一个 { 和最后一个 } 的位置
        start = text.find('{')
        end = text.rfind('}')

        if start != -1 and end != -1:
            return text[start:end + 1]

        raise ValueError("No valid JSON found in response")

    async def call_api(self, batch_size: int = 5) -> List[Dict]:
        """调用API生成一批学生信息"""
        prompt = f"""作为一个中国教育专家，请生成{batch_size}个不同的K12学生信息。每个学生信息需要包含以下字段，并且要确保信息的合理性和多样性。

        必须严格按照以下JSON格式返回：
        {{
            "students": [
                {{
                    "age": <6-17的整数>,
                    "grade": <1-12的整数>,
                    "region": <省份名称>,
                    "school_type": <"重点中学"或"普通中学"等>,
                    "academic_performance": <0-100的整数>,
                    "interests": [兴趣爱好列表],
                    "family_background": {{
                        "parents_education": [父母学历],
                        "education_support": <教育支持程度>,
                        "learning_resources": <学习资源情况>
                    }},
                    "learning_traits": {{
                        "self_motivation": <自主学习动力>,
                        "learning_style": <学习方式>,
                        "attention_span": <注意力水平>
                    }},
                    "additional_education": {{
                        "tutoring": [补课科目],
                        "online_courses": [在线课程],
                        "participation": [竞赛活动]
                    }}
                }},
                ...
            ]
        }}

        生成规则：
        1. 年龄和年级对应关系要合理（小学1-6年级：6-11岁，初中7-9年级：12-14岁，高中10-12年级：15-17岁）
        2. 保持信息的多样性，避免生成过于相似的学生信息
        3. 家庭背景和教育资源要合理匹配
        4. 兴趣爱好和额外教育要符合年龄特点
        5. 学习特征要反映学生的个性化差异
        """

        data = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a professional Chinese education expert."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.8
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, headers=self.headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    return json.loads(self._extract_json_from_text(result['choices'][0]['message']['content']))['students']
                else:
                    raise Exception(f"API Error: {await response.text()}")

    async def generate_student_infos(self, total_students: int, batch_size: int = 5) -> List[Dict]:
        """生成指定数量的不重复学生信息"""
        unique_students = []
        attempts = 0
        max_attempts = total_students * 2  # 设置最大尝试次数

        while len(unique_students) < total_students and attempts < max_attempts:
            try:
                # 获取一批学生信息
                batch = await self.call_api(batch_size)

                # 过滤重复的学生信息
                for student in batch:
                    if not self._is_duplicate(student):
                        unique_students.append(student)
                        print(f"Generated {len(unique_students)}/{total_students} unique students")
                    if len(unique_students) >= total_students:
                        break

                attempts += 1
                await asyncio.sleep(1)  # 添加延迟避免频繁请求

            except Exception as e:
                print(f"Error during generation: {str(e)}")
                await asyncio.sleep(2)

        return unique_students[:total_students]

    def save_to_file(self, students: List[Dict], filename: str = "student_infos.json"):
        """保存学生信息到文件"""
        # 获取当前时间
        now = datetime.now()
        # 格式化日期部分为 YYYYMMDD
        date_str = now.strftime("%Y%m%d")
        # 计算今天已过去的秒数
        seconds_since_midnight = now.hour * 3600 + now.minute * 60 + now.second
        # 拼接成所需格式
        formatted_time = f"{date_str}_{seconds_since_midnight}"
        filename = filename.split(".")[0] + '_' + formatted_time + '.json'

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({"students": students}, f, ensure_ascii=False, indent=2)


async def main():
    api_key = "sk-bCY6IwFu0nbiaRTF528e01387bFf4c3eB2E224B6201129A5"
    generator = StudentInfoGenerator(api_key)

    # 生成100个不重复的学生信息
    students = await generator.generate_student_infos(total_students=1000, batch_size=8)

    # 保存结果
    generator.save_to_file(students)
    print(f"Successfully generated and saved {len(students)} unique student profiles")


if __name__ == "__main__":
    asyncio.run(main())