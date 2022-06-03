import html
import os
import pytest

from mlflow.exceptions import MlflowException
from mlflow.pipelines.cards import BaseCard, CardTab


class ProfileReport:
    def to_html(self):
        return "pandas-profiling"


class FakeCard(BaseCard):
    def __init__(self):
        super().__init__(
            template_root=os.path.join(os.path.dirname(__file__)),
            template_name="fake.html",
            pipeline_name="fake pipeline",
            step_name="fake step",
        )


def test_verify_card_information():
    from markdown import markdown as md_to_html

    profile1 = ProfileReport()
    profile2 = ProfileReport()

    ingest_card = (
        FakeCard()
        .add_markdown("MARKDOWN_1", "#### Hello, world!")
        .add_html("HTML_1", "<span style='color:blue'>blue</span>")
        .add_pandas_profile("Profile 1", profile1)
        .add_pandas_profile("Profile 2", profile2)
        .add_text("1,2,3.")
    )

    profile_iframe = lambda profile: (
        "<iframe srcdoc='"
        + html.escape(profile.to_html())
        + "' width='100%' height='500' frameborder='0'></iframe>"
    )
    for key, value in {
        "MARKDOWN_1": md_to_html("#### Hello, world!"),
        "HTML_1": "<span style='color:blue'>blue</span>",
    }.items():
        assert key in ingest_card._context
        assert ingest_card._context[key] == value

    assert ingest_card._tabs == [
        ("Profile 1", profile_iframe(profile1)),
        ("Profile 2", profile_iframe(profile2)),
    ]
    assert ingest_card._string_builder.getvalue() == "1,2,3."
    assert ingest_card.to_text() == "1,2,3."


def test_card_tab_works():
    tab = (
        CardTab("tab", "{{MARKDOWN_1}}{{HTML_1}}{{PROFILE_1}}")
        .add_html("HTML_1", "<span style='color:blue'>blue</span>")
        .add_markdown("MARKDOWN_1", "#### Hello, world!")
        .add_pandas_profile("PROFILE_1", ProfileReport())
    )
    assert (
        tab.to_html()
        == "<h4>Hello, world!</h4><span style='color:blue'>blue</span>"
        + "<iframe srcdoc='pandas-profiling' width='100%' height='500' frameborder='0'></iframe>"
    )


def test_card_tab_fails_with_invalid_variable():
    with pytest.raises(MlflowException, match=r"(not a valid template variable)"):
        CardTab("tab", "{{MARKDOWN_1}}").add_html("HTML_1", "<span style='color:blue'>blue</span>")
