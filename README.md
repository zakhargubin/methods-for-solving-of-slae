# methods-for-solving-of-slae

## Сборка проекта

Из корня проекта:

```bash
cmake -B build
cmake --build build
```

---

## Запуск всех тестов

```bash
ctest --test-dir build --output-on-failure
```

---

## Запуск benchmark'ов

### Benchmark итерационных методов

```bash
./build/benchmark/benchmark_iterative_solvers
```

После запуска будут созданы csv-файлы с историей и итогами.


## Построение графиков

### Графики для benchmark_iterative_solvers

```bash
python3 benchmark/plot_iterative_benchmark.py
```

Картинки будут сохранены в папку:

```text
benchmark/iterative_plots/
```

---

## Требования

Нужно, чтобы были установлены:

* C++17-совместимый компилятор
* CMake
* Python 3
* matplotlib
