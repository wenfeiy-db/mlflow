import jinja2
import pprint
import os
import html
from jinja2 import meta
from io import StringIO
from IPython.display import HTML, display
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

    def add_markdown(self, name, markdown):
        if name not in self._variables:
            raise ValueError(
                f"{name} is not a valid markdown variable found in template '{self.template_name}'"
            )
        self._context[name] = md_to_html(markdown)
        return self

    def add_pandas_profile(self, name, profile):
        if name not in self._variables:
            raise ValueError(
                f"{name} is not a valid Pandas profile variable found in template "
                f"'{self.template_name}'"
            )

        self._context[name] = (
            "<iframe srcdoc='"
            + html.escape(profile.to_html())
            + "' width='100%' height='500' frameborder='0'></iframe>"
        )

        # WIP: Shadow DOM implementation, require sophisticated escaping.
        # html_lines = profile.to_html().split('\n')

        # # Replacements are to fix: Uncaught SyntaxError: Octal escape sequences errors
        # html_lines = [
        #     "\"" + html.escape(line.replace("\\","\\\\").replace("`",  '\\`')) + "\""
        #     for line in html_lines
        # ]
        # innerHTML = "[" + ", ".join(html_lines) + "].join('\\n')"

        # # Uncaught DOMException: Failed to execute 'define' on 'CustomElementRegistry'
        # # the tag must be all lower case and contain at least one dash
        # tag_name = name.lower().replace("_", "-")

        # self._context[name] = f'''
        # <script>
        #     function decodeHtml(html) {{
        #         var txt = document.createElement("textarea");
        #         txt.innerHTML = html;
        #         return txt.value;
        #     }}
        #     (function () {{
        #         customElements.define('{tag_name}', class extends HTMLElement {{
        #             connectedCallback() {{
        #                 const shadow = this.attachShadow({{mode: 'closed'}});
        #                 shadow.innerHTML = decodeHtml({innerHTML});
        #             }}
        #         }});
        #     }})();
        # </script>
        # <{tag_name}></{tag_name}>
        # '''
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

    def display(self):
        display(HTML(self.to_html()))


class IngestCard(BaseCard):
    def __init__(self):
        super().__init__(
            template_root=os.path.join(os.path.dirname(__file__), "templates"),
            template_name="ingest.html",
        )
