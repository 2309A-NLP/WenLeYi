"""模型包初始化模块

该包包含与目标函数、优化器和网络架构相关的模块。

要添加一个名为 'dummy' 的自定义模型类，你需要：
1. 添加一个名为 'dummy_model.py' 的文件
2. 定义一个继承自 BaseModel 的子类 DummyModel
3. 实现以下五个函数：
   -- <__init__>:                      初始化类；首先调用 BaseModel.__init__(self, opt)
   -- <set_input>:                     从数据集中解包数据并进行预处理
   -- <forward>:                       产生中间结果
   -- <optimize_parameters>:           计算损失、梯度并更新网络权重
   -- <modify_commandline_options>:    （可选）添加模型特定的选项并设置默认选项

在 <__init__> 函数中，你需要定义四个列表：
    -- self.loss_names (str list):          指定要绘制和保存的训练损失
    -- self.model_names (str list):         定义训练中使用的网络
    -- self.visual_names (str list):        指定要显示和保存的图像
    -- self.optimizers (optimizer list):    定义并初始化优化器

现在你可以通过指定参数 '--model dummy' 来使用模型类。
请参考我们的模板模型类 'template_model.py' 了解更多细节。
"""

import importlib                    # 动态导入模块的工具库
from src.face3d.models.base_model import BaseModel  # 导入模型基类


def find_model_using_name(model_name):
    """根据名称动态导入并查找模型类。

    该函数通过动态导入 "face3d.models/[model_name]_model.py" 模块，
    在该模块中查找名为 DatasetNameModel() 的类并实例化。
    该类必须是 BaseModel 的子类，且查找不区分大小写。

    参数:
        model_name (str): 模型名称，例如 'facerecon'

    返回:
        对应的模型类
    """
    # 构建模块文件名，例如 "face3d.models.facerecon_model"
    model_filename = "face3d.models." + model_name + "_model"
    # 动态导入模块
    modellib = importlib.import_module(model_filename)
    model = None
    # 将模型名称去掉下划线并加上 'model' 后缀作为目标类名
    target_model_name = model_name.replace('_', '') + 'model'
    # 遍历模块中所有属性，查找匹配的类
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    # 如果未找到匹配的类，打印错误信息并退出
    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model


def get_option_setter(model_name):
    """返回模型类的静态方法 <modify_commandline_options>。

    该方法用于在命令行解析阶段修改模型相关的参数选项。

    参数:
        model_name (str): 模型名称

    返回:
        对应模型类的 modify_commandline_options 方法
    """
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_model(opt):
    """根据给定的选项创建模型实例。

    该函数封装了模型的创建过程，
    是本包与 'train.py'/'test.py' 之间的主要接口。

    参数:
        opt: 实验选项对象

    返回:
        模型实例

    使用示例:
        >>> from models import create_model
        >>> model = create_model(opt)
    """
    # 根据选项中的模型名称查找对应的模型类
    model = find_model_using_name(opt.model)
    # 实例化模型
    instance = model(opt)
    print("model [%s] was created" % type(instance).__name__)
    return instance
