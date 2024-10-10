import os
import shutil
import subprocess
from typing import Dict, List, Optional

from loguru import logger

from ai_scientist.llm import extract_json_between_markers, get_response_from_llm

PROMPT_SYSTEM = """
Generate latex code for the research literature survey/summary result based on the latex template provided.
Only output a complete latex code, no other text.
In the paper, come up with best title based on the questions in the summary.
"""

PROMPT_GENERATION = """
<summary>
```
{summary}
```
</summary>

<template>
```
{tex_template}
```
</template>
"""


def generate_latex(
    base_dir,
    client,
    model,
    summary: List[Dict],
    prompt_generation: Optional[str] = None,
    prompt_system: Optional[str] = None,
    save_path: str = "summary.tex",
) -> str:
    template_path = os.path.join(base_dir, "latex", "ieeetrans.tex")
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
    save_path = os.path.join(base_dir, "latex", save_path or "summary.tex")
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
    latex_dir = os.path.join(base_dir, "latex")
    latex_path = os.path.join(latex_dir, latex_file)

    logger.debug(latex_dir)
    logger.debug(latex_path)

    # Directory to store the output (same as the LaTeX directory for now)
    output_dir = latex_dir

    # Commands to run to generate the PDF and specify the output directory
    commands = [
        [
            "pdflatex",
            "-interaction=nonstopmode",
            f"-output-directory={output_dir}",
            latex_file,
        ],
        # ["pdflatex", "-interaction=nonstopmode", f"-output-directory={output_dir}", latex_path],
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
