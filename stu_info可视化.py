# visualization.py
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
from matplotlib import font_manager

# 设置中文字体
try:
    # 方案1：使用系统自带的中文字体
    # plt.rcParams['font.family'] = ['sans-serif']
    # plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun', 'FangSong', 'Arial Unicode MS']

    # 方案2：如果有特定字体文件，可以直接添加
    font_path = 'C:/Windows/Fonts/SimHei.ttf'  # 替换为你的字体文件路径
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.sans-serif'] = ['SimHei'] + plt.rcParams['font.sans-serif']

    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 确保seaborn也使用相同的字体设置
    sns.set(font=plt.rcParams['font.sans-serif'][0], font_scale=1)

except Exception as e:
    print(f"Font setting error: {str(e)}")
    print("Available fonts:", font_manager.findSystemFonts())

class StudentInfoVisualizer:
    def __init__(self, json_file: str):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            if "students" in self.data:  # 如果是包含students键的格式
                self.students = self.data["students"]
            else:  # 如果直接是学生列表
                self.students = self.data

    def plot_basic_info_distribution(self):
        """绘制基本信息分布（年龄、年级、成绩）"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), squeeze=False)

        # 年龄分布
        ages = [s['age'] for s in self.students]
        sns.histplot(ages, bins=12, ax=axes[0, 0])
        axes[0, 0].set_title('Age Distribution')

        # 年级分布
        grades = [s['grade'] for s in self.students]
        sns.histplot(grades, bins=12, ax=axes[0, 1])
        axes[0, 1].set_title('Grade Distribution')

        # 成绩分布
        scores = [s['academic_performance'] for s in self.students]
        sns.histplot(scores, bins=20, ax=axes[0, 2])
        axes[0, 2].set_title('Academic Performance Distribution')

        plt.tight_layout()
        plt.show()

    def plot_region_school_distribution(self):
        """绘制地区和学校类型分布"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), squeeze=False)

        # 地区分布
        regions = Counter([s['region'] for s in self.students])
        region_df = pd.DataFrame.from_dict(regions, orient='index', columns=['count'])
        region_df.plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Region Distribution')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # 学校类型分布
        schools = Counter([s['school_type'] for s in self.students])
        axes[0, 1].pie(schools.values(), labels=schools.keys(), autopct='%1.1f%%')
        axes[0, 1].set_title('School Type Distribution')

        plt.tight_layout()
        plt.show()

    def plot_interests_distribution(self):
        """绘制兴趣爱好分布"""
        all_interests = []
        for s in self.students:
            all_interests.extend(s['interests'])
        interests_count = Counter(all_interests)

        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(interests_count.keys()), y=list(interests_count.values()))
        plt.title('Distribution of Interests')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def plot_parents_education(self):
        """绘制父母教育程度分布"""
        father_edu = []
        mother_edu = []
        for s in self.students:
            parents_edu = s['family_background']['parents_education']
            father_edu.append(parents_edu[0])
            mother_edu.append(parents_edu[1])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 父亲教育程度
        sns.countplot(x=father_edu, ax=ax1)
        ax1.set_title("Father's Education")
        ax1.tick_params(axis='x', rotation=45)

        # 母亲教育程度
        sns.countplot(x=mother_edu, ax=ax2)
        ax2.set_title("Mother's Education")
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    def plot_learning_traits(self):
        """绘制学习特征分布"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), squeeze=False)

        # 自主学习动力
        motivation = Counter([s['learning_traits']['self_motivation'] for s in self.students])
        axes[0, 0].pie(motivation.values(), labels=motivation.keys(), autopct='%1.1f%%')
        axes[0, 0].set_title('Self Motivation Distribution')

        # 学习方式
        style = Counter([s['learning_traits']['learning_style'] for s in self.students])
        axes[0, 1].pie(style.values(), labels=style.keys(), autopct='%1.1f%%')
        axes[0, 1].set_title('Learning Style Distribution')

        # 注意力水平
        attention = Counter([s['learning_traits']['attention_span'] for s in self.students])
        axes[0, 2].pie(attention.values(), labels=attention.keys(), autopct='%1.1f%%')
        axes[0, 2].set_title('Attention Span Distribution')

        plt.tight_layout()
        plt.show()

    def plot_additional_education(self):
        """绘制额外教育情况"""
        all_tutoring = []
        all_courses = []
        all_participation = []

        for s in self.students:
            all_tutoring.extend(s['additional_education']['tutoring'])
            all_courses.extend(s['additional_education']['online_courses'])
            all_participation.extend(s['additional_education']['participation'])

        # 修改这里：使用squeeze=False确保返回的是Axes对象数组
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), squeeze=False)
        # 现在axes是2D数组，需要用axes[0,0]这样的形式访问

        # 课外辅导
        tutoring_count = Counter(all_tutoring)
        sns.barplot(x=list(tutoring_count.keys()), y=list(tutoring_count.values()), ax=axes[0, 0])
        axes[0, 0].set_title('Tutoring Subjects')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # 在线课程
        courses_count = Counter(all_courses)
        sns.barplot(x=list(courses_count.keys()), y=list(courses_count.values()), ax=axes[1, 0])
        axes[1, 0].set_title('Online Courses')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # 竞赛活动
        participation_count = Counter(all_participation)
        sns.barplot(x=list(participation_count.keys()), y=list(participation_count.values()), ax=axes[2, 0])
        axes[2, 0].set_title('Competition Participation')
        axes[2, 0].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    def visualize_all(self):
        """生成所有可视化图表"""
        # 设置绘图风格
        sns.set_theme(style="whitegrid")
        self.plot_basic_info_distribution()
        self.plot_region_school_distribution()
        self.plot_interests_distribution()
        self.plot_parents_education()
        self.plot_learning_traits()
        self.plot_additional_education()


# 使用示例
if __name__ == "__main__":
    visualizer = StudentInfoVisualizer("student_infos.json")
    visualizer.visualize_all()