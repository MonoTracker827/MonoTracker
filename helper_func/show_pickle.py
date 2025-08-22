import os, sys, pickle, argparse

code_dir = os.path.dirname(os.path.realpath(__file__))
LOG_DIR = f'{code_dir}/../logs'

def show_pickle(seq_name, f_idx0, f_idx1):
    pickle_folder = f"{LOG_DIR}/{seq_name}/BA_corres/idx0_{f_idx0}_idx1_{f_idx1}"
    if not os.path.exists(pickle_folder):
        raise FileNotFoundError(f"The folder {pickle_folder} does not exist")

    fig1_name_3d = f"{pickle_folder}/before_BA.pickle"
    fig2_name_3d = f"{pickle_folder}/after_BA.pickle"
    fig1_name = f"{pickle_folder}/before_BA_corres.pickle"
    fig2_name = f"{pickle_folder}/after_BA_corres.pickle"
    fig1 = pickle.load(open(fig1_name, "rb"))
    fig2 = pickle.load(open(fig2_name, "rb"))
    fig1_3d = pickle.load(open(fig1_name_3d, "rb"))
    fig2_3d = pickle.load(open(fig2_name_3d, "rb"))
    fig1_3d.show()
    fig2_3d.show()
    fig1.show()
    fig2.show()
    # wait for user to close the figures
    input("Press Enter to continue...")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_name", type=int, default=0)
    parser.add_argument("--idx0", type=int, default=0)
    parser.add_argument("--idx1", type=int, default=1)
    args = parser.parse_args()

    show_pickle(args.seq_name, args.idx0, args.idx1)