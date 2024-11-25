def calculate_cognition(age, region, is_key_school):
    # 年龄认知值映射
    age_mapping = [
        (3, 6, 1),
        (7, 9, 3),
        (10, 12, 5),
        (13, 15, 7),
        (16, 18, 9),
    ]
    # 地区认知倍数映射
    region_mapping = {
        "北京": 1.5, "上海": 1.5, "江苏": 1.5, "天津": 1.5,
        "福建": 1.3, "河南": 1.3, "河北": 1.3, "山西": 1.3,
        "江西": 1.3, "湖北": 1.3, "湖南": 1.3, "广东": 1.3,
        "安徽": 1.3, "山东": 1.3, "重庆": 1.3,
        "甘肃": 1.1, "青海": 1.1, "内蒙古": 1.1, "黑龙江": 1.1,
        "吉林": 1.1, "辽宁": 1.1, "宁夏": 1.1, "海南": 1.1,
        "陕西": 1.1, "四川": 1.1,
        "云南": 1.0, "广西": 1.0, "贵州": 1.0, "新疆": 1.0,
        "西藏": 1.0,
    }

    # 根据年龄确定认知值
    base_cognition = next((value for start, end, value in age_mapping if start <= age <= end), 0)

    # 根据地区倍数调整
    region_multiplier = region_mapping.get(region, 1.0)
    cognition = base_cognition * region_multiplier

    # 根据学校类型调整
    if age >= 10 and is_key_school:
        cognition += 1

    return cognition


# 示例调用
age = 12
region = "北京"
is_key_school = True
print(calculate_cognition(age, region, is_key_school))  # 输出：8.5
