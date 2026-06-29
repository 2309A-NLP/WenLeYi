"""HTML 页面生成模块

该模块实现了 HTML 类，用于将图像和文本保存为单个 HTML 文件。
支持添加标题、图像行和保存操作。
基于 Python 库 'dominate'，该库通过 DOM API 创建和操作 HTML 文档。
"""
import dominate                     # HTML DOM 操作库
from dominate.tags import meta, h3, table, tr, td, p, a, img, br  # HTML 标签
import os                           # 操作系统接口模块


class HTML:
    """HTML 页面生成类。

    该类允许将图像和文本保存到一个 HTML 文件中。
    支持以下功能：
    - <add_header>: 向 HTML 文件添加文本标题
    - <add_images>: 向 HTML 文件添加一行图像
    - <save>: 将 HTML 保存到磁盘

    基于 Python 库 'dominate'，通过 DOM API 创建和操作 HTML 文档。
    """

    def __init__(self, web_dir, title, refresh=0):
        """初始化 HTML 类。

        参数:
            web_dir (str) -- 存储网页的目录。HTML 文件将创建在 <web_dir>/index.html；
                           图像将保存在 <web_dir>/images/
            title (str)   -- 网页标题
            refresh (int) -- 网站自动刷新间隔（秒）；如果为 0 则不刷新
        """
        self.title = title          # 网页标题
        self.web_dir = web_dir      # 网页目录
        self.img_dir = os.path.join(self.web_dir, 'images')  # 图像存储目录
        # 创建必要的目录
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        # 创建 dominate 文档对象
        self.doc = dominate.document(title=title)
        if refresh > 0:
            # 设置页面自动刷新
            with self.doc.head:
                meta(http_equiv="refresh", content=str(refresh))

    def get_image_dir(self):
        """返回存储图像的目录路径。"""
        return self.img_dir

    def add_header(self, text):
        """向 HTML 文件添加标题。

        参数:
            text (str) -- 标题文本
        """
        with self.doc:
            h3(text)  # 添加三级标题

    def add_images(self, ims, txts, links, width=400):
        """向 HTML 文件添加一行图像。

        参数:
            ims (str list)   -- 图像路径列表
            txts (str list)  -- 网页上显示的图像名称列表
            links (str list) -- 超链接列表；点击图像将跳转到对应链接
            width (int)      -- 图像显示宽度（像素），默认 400
        """
        # 创建一个固定布局的表格
        self.t = table(border=1, style="table-layout: fixed;")
        self.doc.add(self.t)
        with self.t:
            with tr():  # 创建表格行
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            # 创建图像链接
                            with a(href=os.path.join('images', link)):
                                img(style="width:%dpx" % width, src=os.path.join('images', im))
                            br()  # 换行
                            p(txt)  # 显示图像名称

    def save(self):
        """将当前内容保存到 HTML 文件。"""
        html_file = '%s/index.html' % self.web_dir
        f = open(html_file, 'wt')
        f.write(self.doc.render())  # 渲染并写入 HTML 内容
        f.close()


if __name__ == '__main__':
    # 使用示例
    html = HTML('web/', 'test_html')  # 创建 HTML 页面
    html.add_header('hello world')    # 添加标题

    # 准备图像数据
    ims, txts, links = [], [], []
    for n in range(4):
        ims.append('image_%d.png' % n)    # 图像文件名
        txts.append('text_%d' % n)        # 显示文本
        links.append('image_%d.png' % n)  # 链接目标
    html.add_images(ims, txts, links)     # 添加图像行
    html.save()                           # 保存 HTML 文件
