from calculate_cog import CognitionCalculator


class PromptGenerator:
    """
    根据认知值和知识点难度生成 Prompt
    """

    @staticmethod
    def generate_prompt(student_cognition: float, knowledge_difficulty: int, knowledge_point: str) -> str:
        difficulty_gap = knowledge_difficulty - student_cognition
        if difficulty_gap < 0:
            return f"""
            学生的认知能力足够理解该知识点，请生成该知识点的教学内容。知识点内容如下：
            "{knowledge_point}"
            要求：
            - 提供详细扎实的讲解
            """
        elif difficulty_gap <= 1:
            return f"""
            学生的认知能力稍低于该知识点，请生成该知识点的教学内容。知识点内容如下：
            "{knowledge_point}"
            要求：
            - 在讲解时加入适当但不要太多的类比帮助学生理解。
            - 难易适当，讲解详细
            """
        elif difficulty_gap <= 2:
            return f"""
            学生的认知能力低于该知识点较多，请生成该知识点的教学内容。知识点内容如下：
            "{knowledge_point}"
            要求：
            - 先提供核心知识的类比。
            - 再分步骤逐步讲解知识点的关键内容。
            """
        else:
            return f"""
            学生的认知能力远低于该知识点。知识点内容如下：
            "{knowledge_point}"
            要求：
            - 不讲解该知识点，而是讲解适合学生当前水平的该知识点的前置知识。
            - 讲解详细且适合该学生认知的前置知识
            """
