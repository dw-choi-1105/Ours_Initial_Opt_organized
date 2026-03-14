"""
YAML 기반 실험 배치 러너

사용법:
  python run_experiments.py                          # 전체 실험 순차 실행
  python run_experiments.py --dry-run                # 명령어만 출력 (실행 안 함)
  python run_experiments.py --skip-existing          # 이미 결과 있는 실험 건너뜀
  python run_experiments.py --only euler_deblur_motion        # 이름으로 선택
  python run_experiments.py --only 0,2               # 인덱스로 선택 (0-based)
  python run_experiments.py --config my_exp.yaml     # 다른 YAML 사용
  python run_experiments.py --workdir-root mydir     # 결과 폴더 루트 지정
"""

import argparse
import datetime
import subprocess
import sys
from pathlib import Path

import yaml


# store_true 플래그 목록 (값 없이 플래그명만 추가되는 인자들)
BOOL_FLAGS = frozenset({
    "efficient_memory",
    "use_grad_checkpoint",
    "use_amp",
    "use_sdo",
    "use_posterior_sampling",
})


def load_config(yaml_path: str):
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    base_config = data.get("base_config", {})
    base_workdir_root = data.get("base_workdir_root", None)
    experiments = data.get("experiments", [])

    # 유효성 검사
    names_seen = set()
    for i, exp in enumerate(experiments):
        for required in ("name", "script", "task"):
            if required not in exp:
                raise ValueError(f"experiments[{i}]: '{required}' 키가 없습니다.")
        name = exp["name"]
        if name in names_seen:
            raise ValueError(f"실험 이름 중복: '{name}'")
        names_seen.add(name)

    return base_config, base_workdir_root, experiments


def merge_config(base: dict, overrides: dict) -> dict:
    merged = dict(base)
    merged.update(overrides or {})
    return merged


def resolve_workdir(merged: dict, root: str | None, name: str, task: str) -> str:
    # overrides에 workdir 직접 지정된 경우
    if "workdir" in merged:
        return str(merged["workdir"])
    # 자동 생성
    if root is None:
        date_str = datetime.date.today().strftime("%y%m%d")
        root = f"workdir_results_{date_str}"
    return str(Path(root) / task / name)


def build_command(script: str, config: dict, workdir: str) -> list:
    cmd = [sys.executable, script]
    cmd += ["--workdir", workdir, "--base_workdir", workdir]
    for key, value in config.items():
        if key in ("workdir", "base_workdir"):
            continue  # 위에서 이미 주입
        if key in BOOL_FLAGS:
            if value:
                cmd.append(f"--{key}")
            # False면 플래그 생략
        else:
            cmd += [f"--{key}", str(value)]
    return cmd


def run_experiment(cmd: list) -> bool:
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0


def select_experiments(experiments: list, only_spec: str | None) -> list:
    if only_spec is None:
        return experiments

    selected = []
    parts = [p.strip() for p in only_spec.split(",")]
    for part in parts:
        # 숫자면 인덱스로 해석
        if part.isdigit():
            idx = int(part)
            if idx >= len(experiments):
                print(f"경고: 인덱스 {idx}는 범위를 벗어납니다 (총 {len(experiments)}개).")
                continue
            selected.append(experiments[idx])
        else:
            # 이름으로 검색
            matches = [e for e in experiments if e["name"] == part]
            if not matches:
                print(f"경고: '{part}' 이름의 실험을 찾을 수 없습니다.")
                continue
            selected.extend(matches)
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="YAML 설정 파일을 읽어 실험을 순차적으로 실행합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", default="experiments.yaml",
        help="실험 설정 YAML 파일 경로 (기본: experiments.yaml)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="실제 실행 없이 명령어만 출력합니다."
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="workdir가 이미 존재하는 실험은 건너뜁니다."
    )
    parser.add_argument(
        "--only",
        help="실행할 실험을 콤마로 구분해 지정합니다. 이름 또는 0-based 인덱스 사용.\n"
             "예: --only euler_deblur_motion  또는  --only 0,2"
    )
    parser.add_argument(
        "--workdir-root",
        help="결과 저장 루트 폴더. YAML의 base_workdir_root를 덮어씁니다."
    )
    args = parser.parse_args()

    # YAML 로드
    try:
        base_config, base_workdir_root, experiments = load_config(args.config)
    except FileNotFoundError:
        print(f"오류: 설정 파일 '{args.config}'를 찾을 수 없습니다.")
        sys.exit(1)
    except ValueError as e:
        print(f"설정 파일 오류: {e}")
        sys.exit(1)

    # CLI 인자로 루트 폴더 덮어쓰기
    effective_root = args.workdir_root or base_workdir_root

    # 실험 선택
    selected = select_experiments(experiments, args.only)
    if not selected:
        print("실행할 실험이 없습니다.")
        sys.exit(0)

    total = len(selected)
    print(f"총 {total}개 실험 {'(dry-run)' if args.dry_run else ''}")
    print("=" * 60)

    results = []
    for i, exp in enumerate(selected, start=1):
        merged = merge_config(base_config, exp.get("overrides", {}))
        workdir = resolve_workdir(merged, effective_root, exp["name"], exp["task"])
        cmd = build_command(exp["script"], merged, workdir)

        print(f"\n[{i}/{total}] {exp['name']}")
        print(f"  script : {exp['script']}")
        print(f"  workdir: {workdir}")
        print(f"  cmd    : {' '.join(cmd)}")

        if args.skip_existing and Path(workdir).exists():
            print("  → SKIP (workdir 이미 존재)")
            results.append((exp["name"], "SKIP"))
            continue

        if args.dry_run:
            results.append((exp["name"], "DRY-RUN"))
            continue

        ok = run_experiment(cmd)
        status = "OK" if ok else "FAILED"
        print(f"  → {status}")
        results.append((exp["name"], status))

    # 최종 요약
    print("\n" + "=" * 60)
    print("실험 결과 요약")
    print("=" * 60)
    for name, status in results:
        print(f"  {status:10s}  {name}")


if __name__ == "__main__":
    main()
