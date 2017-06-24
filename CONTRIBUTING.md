# Contributing

## Development environment and steps

1. Install `tox` either globally or in a virtualenv: `pip install tox`.
2. Click on the "Fork" button at the top-right of the GitHub page.
3. Clone your fork. Example: `git clone git@github.com:dukebody/sklearn-pandas.git`.
4. Create a new branch to work on the issue/feature you want.
5. Hack out your code. To run the tests and `flake8`, just run `tox`. Tests live in the `tests` subfolder.
6. Submit a new PR with your code, indicating in the PR which issue/feature it relates to.

Note: You don't need to install `sklearn-pandas` in your virtualenv to run the tests. `tox` will automatically create multiple virtual environments to run them with multiple package versions.


## Guidelines

- Remember that `sklearn-pandas` does not expect to do everything. Its scope is to serve as an integration layer between `scikit-learn` and `pandas` where needed. If the feature you want to implement adds a lot of complexity to the code, think twice if it is really needed or can be worked around in a few lines.
- Always write tests for any change introduced.
- If the change involves new options or modifies the public interface, modify also the `README` file explaining how to use it. It uses doctests to test the documentation itself.
- If the change is not just cosmetic, add a line to the Changelog section and your name to the Credits section of the `README` file.
