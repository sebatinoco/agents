import matplotlib.pyplot as plt

def plot_metrics(fig, axes, avg_list, std_list, actor_loss, Q_loss, plot_steps = 50):
    
    avg_list = avg_list[::plot_steps]
    std_list = std_list[::plot_steps]
    
    [ax.cla() for ax in axes]

    axes[0].errorbar(range(len(avg_list)), avg_list, std_list, marker = '.', color = 'C0')
    axes[0].set_title('Agent Rewards')
    axes[0].set_ylabel('Avg Reward')

    axes[1].plot(range(len(actor_loss)), actor_loss)
    axes[1].set_title('Actor loss')
    axes[1].set_ylabel('Q(s,a)')

    axes[2].plot(range(len(Q_loss)), Q_loss)
    axes[2].set_title('Q loss')
    axes[2].set_ylabel('MSE Loss')
    
    [ax.grid('on') for ax in axes]
    [ax.set_xlabel('Evaluation Iteration') for ax in axes]
    fig.tight_layout()
    
    plt.pause(0.05)