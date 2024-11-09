# visualization.py
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

# 设置中文字体和主题
sns.set_theme(style='whitegrid', font='SimHei', font_scale=0.8)
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class StudentInfoVisualizer:
    def __init__(self, json_file: str):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            if "students" in self.data:  # 如果是包含students键的格式
                self.students = self.data["students"]
            else:  # 如果直接是学生列表
                self.students = self.data

        # 创建保存图片的目录
        self.img_dir = 'img'
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

    def plot_basic_info_distribution(self):
        """绘制基本信息分布（年龄、年级、成绩）"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)  # 设置高 DPI
        # 年龄分布
        ages = [s['age'] for s in self.students]
        sns.histplot(ages, bins=12, ax=axes[0])
        axes[0].set_title('年龄分布', fontproperties='SimHei')

        # 年级分布
        grades = [s['grade'] for s in self.students]
        sns.histplot(grades, bins=12, ax=axes[1])
        axes[1].set_title('年级分布', fontproperties='SimHei')

        # 成绩分布
        scores = [s['academic_performance'] for s in self.students]
        sns.histplot(scores, bins=20, ax=axes[2])
        axes[2].set_title('学业成绩分布', fontproperties='SimHei')

        plt.tight_layout()

        # 保存图片
        plt.savefig(os.path.join(self.img_dir, 'basic_info_distribution.png'), dpi=300)
        plt.close()

    def plot_region_school_distribution(self):
        """绘制地区和学校类型分布"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
        # 地区分布
        regions = Counter([s['region'] for s in self.students])
        region_df = pd.DataFrame.from_dict(regions, orient='index', columns=['count'])
        region_df.plot(kind='bar', ax=axes[0])
        axes[0].set_title('地区分布', fontproperties='SimHei')
        axes[0].tick_params(axis='x', rotation=90)

        # 学校类型分布
        schools = Counter([s['school_type'] for s in self.students])
        axes[1].pie(
            schools.values(),
            labels=schools.keys(),
            autopct='%1.1f%%',
            textprops={'fontproperties': 'SimHei'}
        )
        axes[1].set_title('学校类型分布', fontproperties='SimHei')

        plt.tight_layout()

        # 保存图片
        plt.savefig(os.path.join(self.img_dir, 'region_school_distribution.png'), dpi=300)
        plt.close()

    def plot_interests_distribution(self):
        """绘制兴趣爱好分布"""
        all_interests = []
        for s in self.students:
            all_interests.extend(s['interests'])
        interests_count = Counter(all_interests)

        plt.figure(figsize=(12, 6), dpi=300)
        sns.barplot(x=list(interests_count.keys()), y=list(interests_count.values()))
        plt.title('兴趣爱好分布', fontproperties='SimHei')
        plt.xticks(rotation=90, ha='right', fontproperties='SimHei')
        plt.tight_layout()

        # 保存图片
        plt.savefig(os.path.join(self.img_dir, 'interests_distribution.png'), dpi=300)
        plt.close()

    def plot_parents_education(self):
        """绘制父母教育程度分布"""
        father_edu = []
        mother_edu = []
        for s in self.students:
            parents_edu = s['family_background']['parents_education']
            father_edu.append(parents_edu[0])
            mother_edu.append(parents_edu[1])

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

        # 父亲教育程度
        sns.countplot(x=father_edu, ax=axes[0])
        axes[0].set_title('父亲教育程度', fontproperties='SimHei')
        axes[0].tick_params(axis='x', rotation=90)
        axes[0].set_xlabel('', fontproperties='SimHei')
        axes[0].set_ylabel('人数', fontproperties='SimHei')

        # 母亲教育程度
        sns.countplot(x=mother_edu, ax=axes[1])
        axes[1].set_title('母亲教育程度', fontproperties='SimHei')
        axes[1].tick_params(axis='x', rotation=90)
        axes[1].set_xlabel('', fontproperties='SimHei')
        axes[1].set_ylabel('人数', fontproperties='SimHei')

        plt.tight_layout()

        # 保存图片
        plt.savefig(os.path.join(self.img_dir, 'parents_education.png'), dpi=300)
        plt.close()

    def plot_learning_traits(self):
        """绘制学习特征分布"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)

        # 自主学习动力
        motivation = Counter([s['learning_traits']['self_motivation'] for s in self.students])
        axes[0].pie(
            motivation.values(),
            labels=motivation.keys(),
            autopct='%1.1f%%',
            textprops={'fontproperties': 'SimHei'}
        )
        axes[0].set_title('自主学习动力分布', fontproperties='SimHei')

        # 学习方式
        style = Counter([s['learning_traits']['learning_style'] for s in self.students])
        axes[1].pie(
            style.values(),
            labels=style.keys(),
            autopct='%1.1f%%',
            textprops={'fontproperties': 'SimHei'}
        )
        axes[1].set_title('学习方式分布', fontproperties='SimHei')

        # 注意力水平
        attention = Counter([s['learning_traits']['attention_span'] for s in self.students])
        axes[2].pie(
            attention.values(),
            labels=attention.keys(),
            autopct='%1.1f%%',
            textprops={'fontproperties': 'SimHei'}
        )
        axes[2].set_title('注意力水平分布', fontproperties='SimHei')

        plt.tight_layout()

        # 保存图片
        plt.savefig(os.path.join(self.img_dir, 'learning_traits.png'), dpi=300)
        plt.close()

    def plot_additional_education(self):
        """绘制额外教育情况"""
        all_tutoring = []
        all_courses = []
        all_participation = []

        for s in self.students:
            all_tutoring.extend(s['additional_education']['tutoring'])
            all_courses.extend(s['additional_education']['online_courses'])
            all_participation.extend(s['additional_education']['participation'])

        fig, axes = plt.subplots(3, 1, figsize=(12, 15), dpi=300)

        # 课外辅导
        tutoring_count = Counter(all_tutoring)
        sns.barplot(x=list(tutoring_count.keys()), y=list(tutoring_count.values()), ax=axes[0])
        axes[0].set_title('课外辅导科目', fontproperties='SimHei')
        axes[0].tick_params(axis='x', rotation=90)
        axes[0].set_xlabel('', fontproperties='SimHei')
        axes[0].set_ylabel('人数', fontproperties='SimHei')

        # 在线课程
        courses_count = Counter(all_courses)
        sns.barplot(x=list(courses_count.keys()), y=list(courses_count.values()), ax=axes[1])
        axes[1].set_title('在线课程学习', fontproperties='SimHei')
        axes[1].tick_params(axis='x', rotation=90)
        axes[1].set_xlabel('', fontproperties='SimHei')
        axes[1].set_ylabel('人数', fontproperties='SimHei')

        # 竞赛活动
        participation_count = Counter(all_participation)
        sns.barplot(x=list(participation_count.keys()), y=list(participation_count.values()), ax=axes[2])
        axes[2].set_title('竞赛活动参与', fontproperties='SimHei')
        axes[2].tick_params(axis='x', rotation=90)
        axes[2].set_xlabel('', fontproperties='SimHei')
        axes[2].set_ylabel('人数', fontproperties='SimHei')

        plt.tight_layout()

        # 保存图片
        plt.savefig(os.path.join(self.img_dir, 'additional_education.png'), dpi=300)
        plt.close()

    def visualize_all(self):
        """生成所有可视化图表"""
        self.plot_basic_info_distribution()
        self.plot_region_school_distribution()
        self.plot_interests_distribution()
        self.plot_parents_education()
        self.plot_learning_traits()
        self.plot_additional_education()
        print(f"所有图片已保存到 {self.img_dir}/ 目录下。")

# 使用示例
if __name__ == "__main__":
    visualizer = StudentInfoVisualizer("student_infos_20241108_49920.json")
    visualizer.visualize_all()
