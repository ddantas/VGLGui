---
id: 2.1
fase: 2
titulo: Criar e rodar test_threadlocal_queue.py
requisitos: [RF-001, RF-003]
depende: 1.1
paralela: true  # pode rodar em paralelo com 2.2
---

# 2.1 — Criar e rodar `test_threadlocal_queue.py`

## Arquivos

- **Criar:** `test_threadlocal_queue.py` (raiz do projeto)

## O que implementar

Script de verificação standalone que confirma RF-001 e RF-003 sem precisar
de um workflow completo. Deve inicializar o contexto OpenCL, criar 2 threads,
capturar os IDs das filas e verificar isolamento.

```python
#!/usr/bin/env python3
"""
Verifica que vl.get_ocl().commandQueue é thread-local (RF-001)
e que finish_queue() funciona sem exceção (RF-003).
"""
import sys
import os
import threading

# ajusta path para rodar da raiz do projeto
sys.path.insert(0, os.path.dirname(__file__))

import vgl_lib as vl

def init_context():
    """Inicializa contexto OpenCL (necessário antes dos testes)."""
    import pyopencl as cl
    vl.vglClInit(device_type=cl.device_type.GPU)

def main():
    init_context()

    results = {}

    def capture(tid):
        q1 = id(vl.get_ocl().commandQueue)
        q2 = id(vl.get_ocl().commandQueue)  # segunda chamada na mesma thread
        results[tid] = (q1, q2)

    t1 = threading.Thread(target=capture, args=("t1",), name="worker-1")
    t2 = threading.Thread(target=capture, args=("t2",), name="worker-2")
    t1.start(); t2.start()
    t1.join(); t2.join()

    # RF-001-A: mesma fila dentro da mesma thread
    assert results["t1"][0] == results["t1"][1], \
        "FALHOU RF-001-A: filas diferentes na mesma thread (t1)"
    assert results["t2"][0] == results["t2"][1], \
        "FALHOU RF-001-A: filas diferentes na mesma thread (t2)"

    # RF-001-B: filas diferentes entre threads
    assert results["t1"][0] != results["t2"][0], \
        "FALHOU RF-001-B: mesma fila em threads diferentes"

    # RF-003: finish_queue() sem exceção
    vl.get_ocl_context().finish_queue()

    # RF-003: thread que nunca usou queue — sem exceção
    def finish_sem_uso():
        vl.get_ocl_context().finish_queue()
    t3 = threading.Thread(target=finish_sem_uso, name="worker-sem-uso")
    t3.start(); t3.join()

    print("OK: todas as verificações passaram")
    print(f"  t1 queue id: {results['t1'][0]}")
    print(f"  t2 queue id: {results['t2'][0]}")

if __name__ == "__main__":
    main()
```

## Padrão de referência

- `vgl_lib/vglClImage.py` — ver como `vglClInit` é chamado em outros scripts

## Acceptance Criteria

- AC-A: Script roda sem exceção: `python3 test_threadlocal_queue.py`
- AC-B: Output contém "OK: todas as verificações passaram"
- AC-C: Os dois IDs de queue impressos são diferentes

## Verificar

```bash
cd /home/joao/Documents/TCC_1/VGLGui
source my_env/bin/activate
python3 test_threadlocal_queue.py
```
