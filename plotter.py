import matplotlib.pyplot as plt
import torch


class TrainingPlotter:

    def __init__(self, X, y, initial_w, pause=0.0):

        self.X = X
        self.y = y
        self.pause = pause

        plt.ion()


        # ====================================================
        # Regression plot
        # ====================================================

        self.fig, self.ax = plt.subplots()

        # all data points
        self.all_points = self.ax.scatter(
            X[:, 0],
            y[:, 0],
            alpha=0.4,
            label="All data"
        )

        # highlighted current batch
        self.batch_points = self.ax.scatter(
            [],
            [],
            alpha=0.4,
            label="Current batch"
        )

        sorted_indices = torch.argsort(X[:, 0])
        self.X_sorted = X[sorted_indices]

        y_pred = self.X_sorted @ initial_w

        self.line, = self.ax.plot(
            self.X_sorted[:, 0],
            y_pred.detach(),
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

        loss_surface = torch.zeros_like(W)

        for i in range(W.shape[0]):

            for j in range(W.shape[1]):

                w_test = torch.tensor([
                    [W[i, j]],
                    [B[i, j]]
                ])

                prediction = X @ w_test

                loss_surface[i, j] = (
                    0.5
                    * torch.linalg.vector_norm(
                        prediction - y,
                        2
                    ) ** 2
                )


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
            label="Loss"
        )


        # ====================================================
        # Parameter path
        # ====================================================

        self.w_history = []
        self.b_history = []

        self.param_line, = self.ax2.plot(
            [],
            [],
            marker="o",
            markersize=0,
            linewidth=1.5,
            label="SGD path"
        )

        self.current_point, = self.ax2.plot(
            [],
            [],
            marker="o",
            markersize=1.5
        )

        self.ax2.legend()


    def update(
        self,
        w,
        loss,
        epoch,
        X_batch,
        y_batch
    ):

        # ====================================================
        # Regression plot
        # ====================================================

        prediction = self.X_sorted @ w

        self.line.set_ydata(
            prediction.detach().flatten()
        )


        # highlight current batch
        batch_points = torch.stack(
            (
                X_batch[:, 0],
                y_batch[:, 0]
            ),
            dim=1
        )

        self.batch_points.set_offsets(
            batch_points.detach().numpy()
        )


        self.ax.set_title(
            f"Epoch {epoch} | Loss: {loss.item():.4f}"
        )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


        # ====================================================
        # Parameter-space plot
        # ====================================================

        weight = w[0, 0].item()
        bias = w[1, 0].item()

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

        self.ax2.set_title(
            f"Parameter path | Loss: {loss.item():.4f}"
        )

        self.fig2.canvas.draw()
        self.fig2.canvas.flush_events()

        plt.pause(self.pause)


    def show(self):

        plt.ioff()
        plt.show()