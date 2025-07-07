from d2l import torch as d2l
import torch

# 训练回归问题时的函数
def train_regression(net , loss , optim, train_iter, test_iter, num_epochs , device , showSVG = False):
    # 训练模型通用函数
    if showSVG:
        animator = d2l.Animator(xlabel='epoch' , xlim = [1 , num_epochs],
                        ylabel='loss' ,
                        legend=['train loss' , 'test loss'])
    for epoch in range(num_epochs):
        net.train()
        train_loss_sum = 0.0
        train_step = 0
        for data in train_iter:
            inputData, target = data
            inputData, target = inputData.to(device), target.to(device)
            optim.zero_grad()
            output = net(inputData)
            l = loss(output, target)
            l.backward()
            optim.step()
            train_loss_sum += l.item()
            train_step += 1
        train_loss = train_loss_sum / train_step
        # print(f'epoch {epoch + 1}, avg train loss {train_loss:.4f}')
        # 测试
        net.eval()
        test_loss_sum = 0.0
        test_step = 0
        with torch.no_grad():
            for data in test_iter:
                inputData, target = data
                inputData, target = inputData.to(device), target.to(device)
                output = net(inputData)
                l = loss(output, target)
                test_loss_sum += l.item()
                test_step += 1
        test_loss = test_loss_sum / test_step
        # print(f'epoch {epoch + 1}, avg test loss {test_loss:.4f}')
        # 绘图
        if showSVG:
            animator.add(epoch + 1, (train_loss, test_loss))
    return net

# 训练分类问题时的函数
def train_classification(net , loss , optim, train_iter, test_iter, num_epochs , device , showSVG = False):
    # 训练模型通用函数
    if showSVG:
        animator = d2l.Animator(xlabel='epoch' , xlim = [1 , num_epochs],
                        ylabel='loss' , ylim = [0 , 1],
                        legend=['train acc' , 'test acc'])
    for epoch in range(num_epochs):
        net.train()
        train_acc_count = 0
        train_acc_sum = 0
        train_step = 0
        for data in train_iter:
            inputData, target = data
            inputData, target = inputData.to(device), target.to(device)
            optim.zero_grad()
            output = net(inputData)
            l = loss(output, target)
            l.backward()
            optim.step()
            train_step += 1
            train_acc_count += (output.argmax(dim=1) == target).sum().item()
            train_acc_sum += output.shape[0]
        train_acc = train_acc_count / train_acc_sum
        # print(f'epoch {epoch + 1}, avg train loss {train_loss:.4f}')
        # 测试
        net.eval()
        test_acc_count = 0
        test_acc_sum = 0
        test_step = 0
        with torch.no_grad():
            for data in test_iter:
                inputData, target = data
                inputData, target = inputData.to(device), target.to(device)
                output = net(inputData)
                # l = loss(output, target)
                test_acc_count += (output.argmax(dim=1) == target).sum().item()
                test_acc_sum += output.shape[0]
        test_acc = test_acc_count / test_acc_sum
        # print(f'epoch {epoch + 1}, avg test loss {test_loss:.4f}')
        # 绘图
        if showSVG:
            animator.add(epoch + 1, (train_acc, test_acc))
    return net