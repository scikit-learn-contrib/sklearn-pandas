import nox

@nox.session
def lint(session):
    session.install('pytest==5.3.5', 'setuptools==45.2',
                    'wheel==0.34.2', 'flake8==3.7.9',
                    'numpy==1.18.1', 'pandas==1.0.5')
    session.install('.')
    session.run('flake8', 'sklearn_pandas/', 'tests')

@nox.session
@nox.parametrize('numpy', ['1.18.1', '1.19.4', '1.20.1'])
@nox.parametrize('scipy', ['1.4.1', '1.5.4', '1.6.0'])
@nox.parametrize('pandas', ['1.0.5', '1.1.4', '1.2.2'])
def tests(session, numpy, scipy, pandas):
    session.install('pytest==5.3.5', 
                    'setuptools==45.2',
                    'wheel==0.34.2',
                    f'numpy=={numpy}',
                    f'scipy=={scipy}',
                    f'pandas=={pandas}'
                    )
    session.install('.')
    session.run('py.test', 'README.rst', 'tests')
