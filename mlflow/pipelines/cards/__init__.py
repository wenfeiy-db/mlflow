import jinja2
import pprint
import os
from jinja2 import meta
from io import StringIO
from IPython.display import display
import ipywidgets as widgets
from markdown import markdown as md_to_html


class BaseCard:
    def __init__(self, template_root, template_name):
        self.template_root = template_root
        self.template_name = template_name

        j2_env = jinja2.Environment()
        with open(os.path.join(template_root, template_name)) as f:
            template = j2_env.parse(f.read())
        self._variables = meta.find_undeclared_variables(template)

        self._context = {}
        self._string_builder = StringIO()
        self._pandas_profiles = []

    def add_markdown(self, name, markdown):
        if name not in self._variables:
            raise ValueError(
                f"{name} is not a valid markdown variable found in template '{self.template_name}'"
            )
        self._context[name] = md_to_html(markdown)
        return self

    def add_pandas_profile(self, name, profile):
        self._pandas_profiles.append((name, profile))
        return self

    def add_artifact(self, name, artifact):
        if name not in self._variables:
            raise ValueError(
                f"{name} is not a valid artifact variable found in template '{self.template_name}'"
            )
        self._context[name] = pprint.pformat(artifact)
        return self

    def add_text(self, text):
        self._string_builder.write(text)
        return self

    def to_html(self):
        j2_env = jinja2.Environment(loader=jinja2.FileSystemLoader(self.template_root))
        return j2_env.get_template(self.template_name).render(self._context)

    def to_text(self):
        return self._string_builder.getvalue()

    def display(self, group_pandas_profiles=True):
        if len(self._pandas_profiles) == 0:
            display(widgets.HTML(self.to_html()))
        else:
            tab = widgets.Tab()
            tab.children = [widgets.HTML(self.to_html())] + [
                profile.widgets for _, profile in self._pandas_profiles
            ]
            titles = ["Summary"] + [name for name, _ in self._pandas_profiles]
            for i in range(len(tab.children)):
                tab.set_title(i, titles[i])
            display(tab)


class IngestCard(BaseCard):
    def __init__(self):
        super().__init__(
            template_root=os.path.join(os.path.dirname(__file__), "templates"),
            template_name="ingest.html",
        )
