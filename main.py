from config import config
from utils import GracefulKiller, log_batch_stats


def main():
    killer = GracefulKiller()
    agent = config.agent(config)
    animate = None
    for i in range(config.iterations):
        observes, actions, advantages, disc_sum_rew = agent.build_train_set(animate)
        log_batch_stats(observes, actions, advantages, disc_sum_rew, agent.logger, i)
        agent.update_policy(observes, actions, advantages)
        agent.update_baseline(observes, disc_sum_rew)
        agent.logger.write(display=True)
        if killer.kill_now:
            if input('Terminate training (y/[n])? ') == 'y':
                break
            if input('Activate animation (a/[d])? ') == 'a':
                animate = True
                print('Animate: ON')
            else:
                animate = False
                print('Animate: OFF')
                agent.env.env.close()
            killer.kill_now = False
    agent.sess.close()
    agent.logger.close()


if __name__ == '__main__':
    main()
