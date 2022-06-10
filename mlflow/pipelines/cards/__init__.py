from __future__ import annotations

import base64
import html
import os
import pathlib
import logging
import random
import string
from io import StringIO
from typing import Union

from mlflow.exceptions import MlflowException, INVALID_PARAMETER_VALUE

CARD_PICKLE_NAME = "card.pkl"
CARD_HTML_NAME = "card.html"

# TODO: Make card save / load including card_resources directory
_CARD_RESOURCE_DIR_NAME = f"{CARD_HTML_NAME}.resources"

_logger = logging.getLogger(__name__)


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

    def add_image(self, name: str, image_file_path: str, width: int = None, height: int = None):
        with open(image_file_path, "rb") as f:
            base64_str = base64.b64encode(f.read()).decode('utf-8')

        image_type = pathlib.Path(image_file_path).suffix[1:]

        width_style = f'width="{width}"' if width else ""
        height_style = f'height="{width}"' if height else ""
        img_html = (f'<img src="data:image/{image_type};base64, {base64_str}" '
                   f"{width_style} {height_style} />")
        self.add_html(name, img_html)

    def add_pandas_profile(self, name: str, profile) -> CardTab:
        """
        Add a new tab representing the provided pandas profile to the card.

        :param name: name of the variable in the Jinja2 template
        :param profile: the pandas profile object
        :return: the updated card instance
        """
        try:
            profile_iframe = (
                "<iframe srcdoc='{src}' width='100%' height='500' frameborder='0'></iframe>"
            ).format(src=html.escape(profile.to_html()))
        except Exception as e:
            profile_iframe = f"Unable to create data profile. Error found:\n{e}"
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
    def __init__(self, pipeline_name: str, step_name: str) -> None:
        """
        BaseCard Constructor

        :param pipeline_name: a string representing name of the pipeline.
        :param step_name: a string representing the name of the step.
        """
        self._pipeline_name = pipeline_name
        self._step_name = step_name
        self._template_name = "base.html"

        self._string_builder = StringIO()
        self._tabs = []

    def add_tab(self, name, html_template) -> CardTab:
        """
        Add a new tab with arbitrary content.

        :param name: a string representing the name of the tab.
        :param html_template: a string representing the HTML template for the card content.
        """
        tab = CardTab(name, html_template)
        self._tabs.append((name, tab))
        return tab

    def get_tab(self, name) -> Union[CardTab, None]:
        """
        Returns an existing tab with the specified name. Returns None if not found.

        :param name: a string representing the name of the tab.
        """
        for key, tab in self._tabs:
            if key == name:
                return tab
        return None

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

        def get_random_id(length=6):
            return "".join(
                random.choice(string.ascii_lowercase + string.digits) for _ in range(length)
            )

        base_template_path = os.path.join(os.path.dirname(__file__), "templates")
        j2_env = jinja2.Environment(loader=jinja2.FileSystemLoader(base_template_path))
        tab_list = [(name, tab.to_html()) for name, tab in self._tabs]
        page_id = get_random_id()
        return j2_env.get_template(self._template_name).render(
            {
                "HEADER_TITLE": f"{self._step_name.capitalize()}@{self._pipeline_name}",
                "TABLINK": f"tablink-{page_id}",
                "CONTENT": f"content-{page_id}",
                "BUTTON_CONTAINER": f"button-container-{page_id}",
                "tab_list": tab_list,
            }
        )

    def to_text(self) -> str:
        """
        :return: the textual representation of the card.
        """
        return self._string_builder.getvalue()

    def save_as_html(self, path) -> None:
        if os.path.isdir(path):
            path = os.path.join(path, CARD_HTML_NAME)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_html())

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

    @staticmethod
    def render_table(table, columns=None):
        """
        Renders a table-like object as an HTML table.

        :param table: Table-like object (e.g. pandas DataFrame, 2D numpy array, list of tuples).
        :param columns: Column names to use. If `table` doesn't have column names, this argument
            provides names for the columns. Otherwise, only the specified columns will be included
            in the output HTML table.
        """
        import pandas as pd
        from pandas.io.formats.style import Styler

        if not isinstance(table, Styler):
            table = pd.DataFrame(table, columns=columns).style

        return (
            table.set_table_attributes('style="border-collapse:collapse"')
            .set_table_styles(
                [
                    {
                        "selector": "table, th, td",
                        "props": [
                            ("border", "1px solid grey"),
                            ("text-align", "left"),
                            ("padding", "5px"),
                        ],
                    }
                ]
            )
            .hide_index()
            .render()
        )


class FailureCard(BaseCard):
    """
    Step card providing information about a failed step execution, including a stacktrace.

    TODO: Migrate the failure card to a tab-based card, removing this class and its associated
          HTML template in the process.
    """

    def __init__(self, pipeline_name: str, step_name: str, failure_traceback: str):
        super().__init__(
            pipeline_name=pipeline_name,
            step_name=step_name,
        )
        self.add_tab("Step Status", "{{ STEP_STATUS }}").add_html(
            "STEP_STATUS",
            '<p><strong>Step status: <span style="color:red">Failed</span></strong></p>',
        )
        self.add_tab("Stacktrace", "{{STACKTRACE}}").add_html(
            "STACKTRACE", f'<p style="margin-top:0px"><code>{failure_traceback}</p></code>'
        )
