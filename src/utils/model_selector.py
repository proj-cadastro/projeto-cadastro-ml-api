import os

def get_best_model_type(campo: str) -> str:
    path = "docs/comparativo.txt"
    if not os.path.exists(path):
        raise ValueError("Relatório comparativo não encontrado.")
    with open(path, encoding="utf-8") as f:
        inside_section = False
        for line in f:
            if line.strip().startswith(f"{campo}:"):
                inside_section = True
            elif inside_section:
                if line.strip().startswith("Melhor modelo:"):
                    return line.split(":")[1].strip()
                if line.strip() == "" or ":" in line:
                    break
    raise ValueError(f"Modelo ideal não encontrado no relatório para {campo}")
