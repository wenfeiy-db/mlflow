from __future__ import annotations
import pprint
import os
from io import StringIO
from typing import Any


class BaseCard:
    def __init__(self, template_root: str, template_name: str) -> None:
        """
        BaseCard Constructor

        :param template_root: a string representing the root directory of the template
        :param template_name: a string representing the file name
        """
        import jinja2
        from jinja2 import meta as jinja2_meta

        self.template_root = template_root
        self.template_name = template_name

        j2_env = jinja2.Environment()
        with open(os.path.join(template_root, template_name)) as f:
            template = j2_env.parse(f.read())
        self._variables = jinja2_meta.find_undeclared_variables(template)

        self._context = {}
        self._string_builder = StringIO()
        self._pandas_profiles = []

    def add_markdown(self, name: str, markdown: str) -> BaseCard:
        """
        This function first converts the given markdown into HTML then fills it into the variable
        declared in the template.

        :param name: name of the variable in the Jinja2 template
        :param markdown: the markdown content
        :return: the updated card instance
        """
        from markdown import markdown as md_to_html

        if name not in self._variables:
            raise ValueError(
                f"{name} is not a valid markdown variable found in template '{self.template_name}'"
            )
        self._context[name] = md_to_html(markdown)
        return self

    def add_pandas_profile(self, name: str, profile) -> BaseCard:
        """
        Add a new tab representing the provided pandas profile to the card.

        :param name: name of the variable in the Jinja2 template
        :param profile: the pandas profile object
        :return: the updated card instance
        """
        self._pandas_profiles.append((name, profile))
        return self

    def add_artifact(self, name: str, artifact: Any, artifact_format=None) -> BaseCard:
        """
        Add an artifact to the card.

        :param name: name of the variable in the Jinja2 template
        :param artifact: an artifact
        :return: the updated card instance
        """
        if name not in self._variables:
            raise ValueError(
                f"{name} is not a valid artifact variable found in template '{self.template_name}'"
            )
        if not artifact_format:
            artifact = pprint.pformat(artifact)
        self._context[name] = artifact
        return self

    def add_text(self, text: str) -> BaseCard:
        """
        Add text to the textual representation of this card.

        :param text: a string text
        :return: the updated card instance
        """
        self._string_builder.write(text)
        return self

    def to_html(self) -> str:
        """
        This funtion renders the Jinja2 template based on the provided context so far.

        :return: a HTML string
        """
        import jinja2

        j2_env = jinja2.Environment(loader=jinja2.FileSystemLoader(self.template_root))
        return j2_env.get_template(self.template_name).render(self._context)

    def to_text(self) -> str:
        """
        :return: the textual representation of the card.
        """
        return self._string_builder.getvalue()

    def display(self) -> None:
        """
        Display the rendered card as a ipywidget
        """
        import ipywidgets as widgets
        from IPython.display import display

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
