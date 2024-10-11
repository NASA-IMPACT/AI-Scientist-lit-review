import os
import shutil
import subprocess
from typing import Dict, List, Optional

from loguru import logger

from ai_scientist.llm import extract_json_between_markers, get_response_from_llm

PROMPT_SYSTEM = """
You will be provided a literature research survey JSON with questions, answers and contexts. Generate latex code for the research result based on the latex template provided.
Only output a complete latex code, no other text. Be as detailed as you can!
- Reuse the inline citations (using \cite{}) present in the `answer` section and use it within any paragraphs in the latex sections.
- Come up with best title based on the questions in the JSON.
- You are free to augment the section structure of the latex as well.
- Strictly adhere to the answer and context in the JSON.
- Make full use of all the answers and the contexts in the JSON.

Be as much detailed as possible!
"""

PROMPT_GENERATION = """
<literature__json>
```
{summary}
```
</literature_json>

<template_latex>
```
{tex_template}
```
</template_latex>
"""


def generate_latex(
    base_dir,
    client,
    model,
    summary: List[Dict],
    prompt_generation: Optional[str] = None,
    prompt_system: Optional[str] = None,
    template_file: str = "template.tex",
    save_path: str = "summary.tex",
) -> str:
    template_path = os.path.join(base_dir, "latex", template_file)
    template = ""
    with open(template_path) as f:
        template = f.read()

    prompt_generation = (
        (prompt_generation or PROMPT_GENERATION)
        .format(summary=summary, tex_template=template)
        .strip()
    )
    prompt_system = prompt_system or PROMPT_SYSTEM
    response, history = get_response_from_llm(
        prompt_generation,
        client,
        model,
        prompt_system,
    )

    response = response.strip("`")
    save_path = os.path.join(base_dir, "latex", "output/", save_path or "summary.tex")
    logger.info(f"Saving latex code to {save_path}")
    with open(save_path, "w") as f:
        f.write(response)
    return response


def compile_latex(
    base_dir,
    latex_file: str = "summary.tex",
    pdf_output: str = "summary.pdf",
    timeout: int = 30,
):
    """
    Compiles a LaTeX file into a PDF.

    Args:
        base_dir (str): Base directory containing the LaTeX folder.
        latex_file (str): Name of the LaTeX file (e.g., "summary.tex").
        pdf_output (str): Path to save the generated PDF (e.g., "output.pdf").
        timeout (int): Timeout for each compilation step in seconds.

    Returns:
        bool: True if PDF is successfully generated, False otherwise.
    """

    def _compile(
        latex_file,
        pdf_output,
    ):
        latex_dir = os.path.join(base_dir, "latex")
        latex_path = os.path.join(latex_dir, "output", latex_file)

        logger.info(f"Compiling latex | {latex_path}")

        # Directory to store the output (same as the LaTeX directory for now)
        output_dir = os.path.join(latex_dir, "output")

        output_dir = os.path.abspath(output_dir)
        latex_path = os.path.abspath(latex_path)

        logger.debug(latex_dir)
        logger.debug(latex_path)

        # Commands to run to generate the PDF and specify the output directory
        commands = [
            [
                "pdflatex",
                "-interaction=nonstopmode",
                f"-output-directory={output_dir}",
                latex_path,
            ],
            [
                "pdflatex",
                "-interaction=nonstopmode",
                f"-output-directory={output_dir}",
                latex_path,
            ],
        ]

        logger.debug(commands)

        try:
            # Run each command sequentially in the LaTeX file's directory
            for command in commands:
                result = subprocess.run(
                    command,
                    cwd=latex_dir,  # Ensure it runs in the LaTeX directory
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout,
                )
                print("Running command:", " ".join(command))
                print("Output:\n", result.stdout)
                print("Errors:\n", result.stderr)

                if result.returncode != 0:
                    print(f"Error running command {' '.join(command)}.")
                    return False

            # Move the generated PDF to the desired location
            generated_pdf = os.path.join(output_dir, latex_file.replace(".tex", ".pdf"))
            shutil.move(generated_pdf, pdf_output)
            print(f"PDF generated successfully at {pdf_output}")
            return True

        except subprocess.TimeoutExpired:
            print(f"Command {' '.join(command)} timed out after {timeout} seconds")
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {e}")
        except FileNotFoundError:
            print("PDF file not found.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        return False

    _ = _compile(latex_file, pdf_output)
    return _compile(latex_file, pdf_output)
