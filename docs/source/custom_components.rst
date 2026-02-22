Custom Components
=================

This is a guide on how to build a simple app and custom component spec and
launch it via two different schedulers.

See the :doc:`quickstart` for installation and basic usage.

Builtins
--------

Before writing a custom component, check if any of the builtin components
satisfy your needs. TorchX provides a number of builtin components with premade
images. You can discover them via:

.. code:: shell-session

    $ torchx builtins

You can use these either from the CLI, from a pipeline or programmatically like
you would any other component.

.. code:: shell-session

    $ torchx run utils.echo --msg "Hello :)"

Hello World
-----------

Lets start off with writing a simple "Hello World" python app. This is just a
normal python program and can contain anything you'd like.

First, create ``my_app.py``:

.. code-block:: python

    import sys
    import argparse

    def main(user: str) -> None:
        print(f"Hello, {user}!")

    if __name__ == "__main__":
        parser = argparse.ArgumentParser(
            description="Hello world app"
        )
        parser.add_argument(
            "--user",
            type=str,
            help="the person to greet",
            required=True,
        )
        args = parser.parse_args(sys.argv[1:])

        main(args.user)

Now that we have an app we can write the component file for it. This function
allows us to reuse and share our app in a user friendly way.

We can use this component from the ``torchx`` cli or programmatically as part of a
pipeline.

Create ``my_component.py``:

.. code-block:: python

    import torchx.specs as specs

    def greet(user: str, image: str = "my_app:latest") -> specs.AppDef:
        return specs.AppDef(
            name="hello_world",
            roles=[
                specs.Role(
                    name="greeter",
                    image=image,
                    entrypoint="python",
                    args=[
                        "-m", "my_app",
                        "--user", user,
                    ],
                )
            ],
        )

We can execute our component via ``torchx run``. The ``local_cwd`` scheduler
executes the component relative to the current directory.

.. code:: shell-session

    $ torchx run --scheduler local_cwd my_component.py:greet --user "your name"

If we want to run in other environments, we can build a Docker container so we
can run our component in Docker enabled environments such as Kubernetes or via
the local Docker scheduler.

.. note::

    This requires Docker installed and won't work in environments such as Google
    Colab. If you have not done so already follow the install instructions on:
    https://docs.docker.com/get-docker/

Create ``Dockerfile.custom``:

.. code-block:: dockerfile

    FROM ghcr.io/pytorch/torchx:0.1.0rc1

    ADD my_app.py .

Once we have the Dockerfile created we can create our docker image.

.. code:: shell-session

    $ docker build -t my_app:latest -f Dockerfile.custom .

We can then launch it on the local scheduler.

.. code:: shell-session

    $ torchx run --scheduler local_docker my_component.py:greet --image "my_app:latest" --user "your name"

If you have a Kubernetes cluster you can use the
:doc:`Kubernetes scheduler <schedulers/kubernetes>` to launch this on the cluster
instead.

.. code:: shell-session

    $ docker push my_app:latest
    $ torchx run --scheduler kubernetes my_component.py:greet --image "my_app:latest" --user "your name"
