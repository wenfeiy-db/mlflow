import os
from mlflow.pipelines.cards import BaseCard


class RegisterCard(BaseCard):
    def __init__(self):
        super().__init__(
            template_root=os.path.join(os.path.dirname(__file__), "../resources"),
            template_name="register.html",
        )