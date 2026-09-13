import matplotlib.pyplot as plt
import torch


class TrainingPlotter:

    def __init__(self, X, y, model, pause=0.001):

        self.X = X.detach().cpu()
        self.y = y.detach().cpu()
        self.pause = pause

        plt.ion()


        # ====================================================
        # Get initial parameters
        # ====================================================

        weight, bias = self._get_parameters(model)


        # ====================================================
        # Regression plot
        # ====================================================

        self.fig, self.ax = plt.subplots()

        self.ax.scatter(
            self.X[:, 0],
            self.y[:, 0],
            label="Data"
        )

        sorted_indices = torch.argsort(self.X[:, 0])

        self.X_sorted = self.X[sorted_indices]

        y_pred = (
            self.X_sorted[:, 0] * weight
            + bias
        )

        self.line, = self.ax.plot(
            self.X_sorted[:, 0],
            y_pred,
            label="Prediction"
        )

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.legend()


        # ====================================================
        # Parameter-space plot
        # ====================================================

        self.fig2, self.ax2 = plt.subplots()

        self.ax2.set_xlabel("Weight")
        self.ax2.set_ylabel("Bias")
        self.ax2.set_title("Parameter path")


        # ====================================================
        # Loss landscape
        # ====================================================

        weight_values = torch.linspace(
            -0.5,
            1.5,
            200
        )

        bias_values = torch.linspace(
            -0.5,
            1.5,
            200
        )

        W, B = torch.meshgrid(
            weight_values,
            bias_values,
            indexing="xy"
        )


        # ====================================================
        # Vectorized MSE loss calculation
        # ====================================================

        x = self.X[:, 0].reshape(1, 1, -1)
        target = self.y[:, 0].reshape(1, 1, -1)

        prediction = (
            W.unsqueeze(-1) * x
            + B.unsqueeze(-1)
        )

        loss_surface = (
            (prediction - target) ** 2
        ).mean(dim=-1)


        # ====================================================
        # Draw loss landscape
        # ====================================================

        contour = self.ax2.contourf(
            W.numpy(),
            B.numpy(),
            loss_surface.numpy(),
            levels=50,
            cmap="viridis"
        )

        self.ax2.contour(
            W.numpy(),
            B.numpy(),
            loss_surface.numpy(),
            levels=20,
            colors="black",
            alpha=0.25,
            linewidths=0.5
        )

        self.fig2.colorbar(
            contour,
            ax=self.ax2,
            label="MSE Loss"
        )


        # ====================================================
        # Parameter path
        # ====================================================

        self.w_history = []
        self.b_history = []

        self.param_line, = self.ax2.plot(
            [],
            [],
            linewidth=1.5,
            label="SGD path"
        )

        self.current_point, = self.ax2.plot(
            [],
            [],
            marker="o",
            markersize=4
        )

        self.ax2.legend()


    # ========================================================
    # Extract weight and bias from nn.Linear
    # ========================================================

    def _get_parameters(self, model):

        linear = model.linear_stack[0]

        weight = linear.weight.detach().cpu().item()
        bias = linear.bias.detach().cpu().item()

        return weight, bias


    # ========================================================
    # Live update
    # ========================================================

    def update(self, model):

        weight, bias = self._get_parameters(model)


        # ====================================================
        # Regression plot
        # ====================================================

        prediction = (
            self.X_sorted[:, 0] * weight
            + bias
        )

        self.line.set_ydata(
            prediction
        )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


        # ====================================================
        # Parameter-space plot
        # ====================================================

        self.w_history.append(weight)
        self.b_history.append(bias)

        self.param_line.set_data(
            self.w_history,
            self.b_history
        )

        self.current_point.set_data(
            [weight],
            [bias]
        )

        self.fig2.canvas.draw()
        self.fig2.canvas.flush_events()

        plt.pause(self.pause)


    # ========================================================
    # Keep plots open when training finishes
    # ========================================================

    def show(self):

        plt.ioff()
        plt.show()