import html
import os
from mlflow.pipelines.cards import BaseCard


class ProfileReport:
    def to_html(self):
        return "pandas-profiling"


class FakeCard(BaseCard):
    def __init__(self):
        super().__init__(
            template_root=os.path.join(os.path.dirname(__file__)),
            template_name="fake.html",
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
    assert ingest_card._context == {
        "MARKDOWN_1": md_to_html("#### Hello, world!"),
        "HTML_1": "<span style='color:blue'>blue</span>",
    }
    assert ingest_card._pandas_profiles == [
        ("Profile 1", profile_iframe(profile1)),
        ("Profile 2", profile_iframe(profile2)),
    ]
    assert ingest_card._string_builder.getvalue() == "1,2,3."
    assert ingest_card.to_text() == "1,2,3."
