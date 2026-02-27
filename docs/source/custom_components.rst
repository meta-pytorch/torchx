Custom Components
=================

.. tip::

   This guide covers writing a custom :term:`component <Component>` (a Python
   function returning an :term:`AppDef`) and launching it on both local and
   container-based :term:`schedulers <Scheduler>`.
   Check the :doc:`builtins <components/overview>` first -- you may not need a
   custom component.

**Prerequisites:** :doc:`quickstart` (installation) and :doc:`basics` (core concepts).

Builtins
--------

Discover available builtins with ``torchx builtins``:

.. code:: shell-session

    $ torchx run utils.echo --msg "Hello :)"

Hello World
-----------

Create ``my_app.py``:

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

Next, write a component -- a factory function that returns an ``AppDef``.

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

Run the component with the ``local_cwd`` scheduler (executes in the current
directory):

.. code:: shell-session

    $ torchx run --scheduler local_cwd my_component.py:greet --user "your name"

The same launch works from Python using the :py:class:`~torchx.runner.Runner`:

.. code-block:: python

    from torchx.runner import get_runner

    with get_runner() as runner:
        # reference the file:function component, same as the CLI
        app_handle = runner.run_component(
            "my_component.py:greet",
            ["--user", "your name"],
            scheduler="local_cwd",
        )

        # alternatively, call the component function directly
        from my_component import greet

        app = greet(user="your name")
        app_handle = runner.run(app, scheduler="local_cwd")

For container-based schedulers, build a Docker image:

.. note::

   Requires Docker: https://docs.docker.com/get-docker/

Create ``Dockerfile.custom``:

.. code-block:: dockerfile

    FROM ghcr.io/pytorch/torchx:latest

    ADD my_app.py .

Build the image:

.. code:: shell-session

    $ docker build -t my_app:latest -f Dockerfile.custom .

Launch on the local Docker scheduler:

.. code:: shell-session

    $ torchx run --scheduler local_docker my_component.py:greet --image "my_app:latest" --user "your name"

Or push and launch on a :doc:`Kubernetes <schedulers/kubernetes>` cluster:

.. code:: shell-session

    $ docker push my_app:latest
    $ torchx run --scheduler kubernetes my_component.py:greet --image "my_app:latest" --user "your name"

.. seealso::

   :doc:`api_reference`
      Single-page reference with imports, types, and copy-pasteable recipes.

   :doc:`component_best_practices`
      Best practices for entrypoints, simplicity, named resources, and testing.

   :doc:`advanced`
      Registering custom components as CLI builtins via entry points.

   :doc:`components/overview`
      Browse the builtin component library.
