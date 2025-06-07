import argparse


parser = argparse.ArgumentParser()


# 一般設定
parser.add_argument("--algorithm", type = str, default = "REDQ-SMR-Ours", help = "演算法名稱")


# 存檔設定
parser.add_argument("--save_result", action = "store_true", help = "是否存檔")
parser.add_argument("--output_path", type = str, default = "Result", help = "輸出檔案路徑")


# 實驗環境設定
parser.add_argument("--env_name", type = str, default = "HalfCheetah-v5", help = "任務名稱")
parser.add_argument("--device", type = str, default = "cuda:0", help = "實驗使用的設備")
parser.add_argument("--seed", type = int, default = 0, help = "亂數種子")
parser.add_argument("--max_steps", type = int, default = int(3e5), help = "實驗多少步")


# 性能測試設定
parser.add_argument("--test_performance_freq", type = int, default = 1000, help = "每與環境互動多少 steps 要測試一次 actor 性能")
parser.add_argument("--test_n", type = int, default = 20, help = "每次測試 actor 要玩幾局")


# RL設定
parser.add_argument("--replay_buffer_size", type = int, default = int(1e6), help = "replay buffer 的最大空間")
parser.add_argument("--batch_size", type = int, default = 256, help = "random mini-batch size")
parser.add_argument("--start_steps", type = int, default = 5000, help = "最開始使用隨機 action 進行探索")
parser.add_argument("--gamma", type = float, default = 0.99, help = "TD 的 discount")


# REDQ設定
parser.add_argument("--actor_learning_rate", type = float, default = 3e-4, help = "actor 的學習率")
parser.add_argument("--critic_learning_rate", type = float, default = 3e-4, help = "critic 的學習率")
parser.add_argument("--tau", type = float, default = 0.005, help = "以移動平均更新 target 的比例")
parser.add_argument("--alpha", type = float, default = 0.2, help = "entropy 的係數")
parser.add_argument("--adaptive_alpha", type = str, default = "True", choices = ["True", "False"], help = "是否自動調整 alpha")
parser.add_argument("--mbpo_target_entropy", type = str, default = "True", choices = ["True", "False"], help = "是否使用 mbpo 的 target entropy")
parser.add_argument("--critic_size", type = int, default = 10, help = "critic 的數量")
parser.add_argument("--subset_size", type = int, default = 2, help = "計算 target_Q 所使用的 critic target 數量")


# SMR設定
parser.add_argument("--smr_ratio", type = int, default = 10, help = "SMR 重複訓練的比例")


args = parser.parse_args()


if args.env_name[ : -3] == "Humanoid":
    args.alpha = 0.05


if args.adaptive_alpha == "True":
    args.adaptive_alpha = True
else:
    args.adaptive_alpha = False


if args.mbpo_target_entropy == "True":
    args.mbpo_target_entropy = True
else:
    args.mbpo_target_entropy = False


