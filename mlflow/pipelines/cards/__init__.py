from __future__ import annotations

import html
import os
import shutil
from io import StringIO

from mlflow.exceptions import MlflowException, INVALID_PARAMETER_VALUE

CARD_PICKLE_NAME = "card.pkl"
CARD_HTML_NAME = "card.html"

# TODO: Make card save / load including card_resources directory
_CARD_RESOURCE_DIR_NAME = f"{CARD_HTML_NAME}.resources"


class CardTab:
    def __init__(self, name: str, template: str) -> None:
        """
        Construct a step card tab with supported HTML template.

        :param name: a string representing the name of the tab.
        :param template: a string representing the HTML template for the card content.
        """
        import jinja2
        from jinja2 import meta as jinja2_meta

        self.name = name
        self.template = template

        j2_env = jinja2.Environment()
        self._variables = jinja2_meta.find_undeclared_variables(j2_env.parse(template))
        self._context = {}

    def add_html(self, name: str, html_content: str) -> CardTab:
        """
        Adds html to the CardTab.

        :param name: String, name of the variable in the Jinja2 template
        :param html_content: String, the html to replace the named template variable
        :return: the updated card instance
        """
        if name not in self._variables:
            raise MlflowException(
                f"{name} is not a valid template variable defined in template: '{self.template}'",
                error_code=INVALID_PARAMETER_VALUE,
            )
        self._context[name] = html_content
        return self

    def add_markdown(self, name: str, markdown: str) -> CardTab:
        """
        Adds markdown to the card replacing the variable name in the CardTab template.

        :param name: name of the variable in the CardTab Jinja2 template
        :param markdown: the markdown content
        :return: the updated card tab instance
        """
        from markdown import markdown as md_to_html

        self.add_html(name, md_to_html(markdown))
        return self

    def add_pandas_profile(self, name: str, profile) -> CardTab:
        """
        Add a new tab representing the provided pandas profile to the card.

        :param name: name of the variable in the Jinja2 template
        :param profile: the pandas profile object
        :return: the updated card instance
        """
        profile_iframe = (
            "<iframe srcdoc='{src}' width='100%' height='500' frameborder='0'></iframe>"
        ).format(src=html.escape(profile.to_html()))
        self.add_html(name, profile_iframe)
        return self

    def to_html(self) -> str:
        """
        Returns a rendered HTML representing the content of the tab.

        :return: a HTML string
        """
        import jinja2

        j2_env = jinja2.Environment(loader=jinja2.BaseLoader()).from_string(self.template)
        return j2_env.render({**self._context})


class BaseCard:
    def __init__(
        self, template_root: str, template_name: str, pipeline_name: str, step_name: str
    ) -> None:
        """
        BaseCard Constructor

        :param template_root: a string representing the root directory of the template
        :param template_name: a string representing the file name
        """
        import jinja2
        from jinja2 import meta as jinja2_meta

        self.template_root = template_root
        self.template_name = template_name
        self.pipeline_name = pipeline_name
        self.step_name = step_name

        j2_env = jinja2.Environment()
        with open(os.path.join(template_root, template_name)) as f:
            template = j2_env.parse(f.read())
        self._variables = jinja2_meta.find_undeclared_variables(template)

        self._context = {}
        self._string_builder = StringIO()
        self._tabs = []
        self._resource_files = {}

        self.add_html(
            name="HEADER_TITLE",
            html=f"{self.step_name.capitalize()}@{self.pipeline_name}",
        )
        self.add_html(
            name="PAGE_TITLE",
            html=f"MLflow Pipeline {self.step_name.capitalize()}@{self.pipeline_name}",
        )

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
            raise MlflowException(
                f"{name} is not a valid markdown variable found in template '{self.template_name}'",
                error_code=INVALID_PARAMETER_VALUE,
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
        profile_iframe = (
            "<iframe srcdoc='"
            + html.escape(profile.to_html())
            + "' width='100%' height='500' frameborder='0'></iframe>"
        )
        self._tabs.append((name, profile_iframe))
        return self

    def add_tab(self, name, html_template) -> CardTab:
        """
        Add a new tab with arbitrary content.

        :param name: a string representing the name of the tab.
        :param html_template: a string representing the HTML template for the card content.
        """
        tab = CardTab(name, html_template)
        self._tabs.append((name, tab))
        return tab

    def add_html(self, name: str, html: str) -> BaseCard:
        """
        Adds html to the card.

        :param name: name of the variable in the Jinja2 template
        :param html: the html with which to replace the specified template variable
        :return: the updated card instance
        """
        if name not in self._variables:
            raise ValueError(
                f"{name} is not a valid artifact variable found in template '{self.template_name}'"
            )
        self._context[name] = html
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

        baseTemplatePath = os.path.join(os.path.dirname(__file__), "templates")
        j2_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader([self.template_root, baseTemplatePath])
        )
        tab_list = [
            (name, tab.to_html() if isinstance(tab, CardTab) else tab) for name, tab in self._tabs
        ]
        return j2_env.get_template(self.template_name).render(
            {**self._context, "tab_list": tab_list}
        )

    def to_text(self) -> str:
        """
        :return: the textual representation of the card.
        """
        return self._string_builder.getvalue()

    def display(self) -> None:
        """
        Display the rendered card as a ipywidget
        """
        from IPython.display import display, HTML

        display(HTML(self.to_html()))

    def _add_resource_file(self, path):
        """
        Add a resource file. Return a relative path pointing to the file.
        In html content, use the returned relative path instead of original path.
        When calling `save_as_html`, resource files will be saved to the
        `_CARD_RESOURCE_DIR_NAME` subdirectory in the same directory.
        This is a private method and it might change in future.

        TODO: Make pickling / unpickling support resource files.
        """
        res_name = f"r{len(self._resource_files) + 1}_{os.path.basename(path)}"
        rel_path = os.path.join(_CARD_RESOURCE_DIR_NAME, res_name)
        self._resource_files[path] = rel_path
        return rel_path

    def save_as_html(self, path) -> None:
        if os.path.isdir(path):
            path = os.path.join(path, CARD_HTML_NAME)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_html())

        if len(self._resource_files) > 0:
            dir_path = os.path.dirname(path)
            resource_dir = os.path.join(dir_path, _CARD_RESOURCE_DIR_NAME)
            os.makedirs(resource_dir, exist_ok=True)
            for original_path, rel_path in self._resource_files.items():
                dest_path = os.path.join(resource_dir, os.path.basename(rel_path))
                shutil.copy(original_path, dest_path)

    def save(self, path: str) -> None:
        if os.path.isdir(path):
            path = os.path.join(path, CARD_PICKLE_NAME)
        with open(path, "wb") as out:
            import pickle

            pickle.dump(self, out)

    @staticmethod
    def load(path):
        if os.path.isdir(path):
            path = os.path.join(path, CARD_PICKLE_NAME)
        with open(path, "rb") as f:
            import pickle

            return pickle.load(f)


class FailureCard(BaseCard):
    def __init__(self, pipeline_name: str, step_name: str, failure_traceback: str):
        super().__init__(
            template_root=os.path.join(os.path.dirname(__file__), "templates"),
            template_name="failure.html",
            pipeline_name=pipeline_name,
            step_name=step_name,
        )
        self.add_html(
            "STEP_STATUS",
            '<p><strong>Step status: <span style="color:red">Failed</span></strong></p>',
        )
        self.add_html(
            "STACKTRACE", f'<p style="margin-top:0px"><code>{failure_traceback}</p></code>'
        )
