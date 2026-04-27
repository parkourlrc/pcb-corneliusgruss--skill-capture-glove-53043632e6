"""
PIP joint force analysis — built step by step.

Step 1: Just the finger, no forces. Verify geometry.
Step 2: Add the grip force at the tip. Verify direction.
Step 3: Add the axle reaction. Verify equilibrium.
Step 4: Show where the magnet displaces.

Coordinate frame (looking down at right hand from above):
  X = forward (toward fingertip)  = RIGHT on screen
  Y = left (toward thumb/object)  = UP on screen
  Z = up (out of page)            = OUT OF SCREEN

PIP axle is in Z (vertical pin). Finger flexes in XY plane.

Usage:
    python -m glove.scripts.visualize_joint_forces
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Arc
from matplotlib.widgets import Button


class StepByStep:
    def __init__(self):
        self.step = 0
        self.max_step = 4

        # --- Fixed parameters ---
        self.link_len = 22.0       # mm, distal phalanx
        self.joint_angle_deg = 40  # flexion
        self.grip_force = 5.0      # N, pinch grip

        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 10), facecolor="#111")
        self.ax.set_facecolor("#1a1a2e")

        # Next button
        ax_btn = self.fig.add_axes([0.75, 0.02, 0.15, 0.05])
        self.btn_next = Button(ax_btn, "Next step →", color="#333", hovercolor="#555")
        self.btn_next.label.set_color("white")
        self.btn_next.on_clicked(self.next_step)

        # Back button
        ax_back = self.fig.add_axes([0.58, 0.02, 0.15, 0.05])
        self.btn_back = Button(ax_back, "← Back", color="#333", hovercolor="#555")
        self.btn_back.label.set_color("white")
        self.btn_back.on_clicked(self.prev_step)

        self.draw()

    def next_step(self, _=None):
        if self.step < self.max_step:
            self.step += 1
            self.draw()

    def prev_step(self, _=None):
        if self.step > 0:
            self.step -= 1
            self.draw()

    def arrow(self, origin, vec, color, label, label_offset=(0, 0), scale=1.0, lw=2.5):
        """Draw a force arrow with label. vec is in mm (for geometry) or mm-equivalent (for forces)."""
        end = origin + vec * scale
        self.ax.annotate(
            "", xy=end, xytext=origin,
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=15),
        )
        mid = (origin + end) / 2
        self.ax.text(
            mid[0] + label_offset[0], mid[1] + label_offset[1], label,
            color=color, fontsize=10, fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="#1a1a2e", ec=color, alpha=0.9, lw=1),
        )

    def draw(self):
        ax = self.ax
        ax.clear()
        ax.set_facecolor("#1a1a2e")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.08, color="white")
        ax.tick_params(colors="#aaa")
        for sp in ax.spines.values():
            sp.set_color("#333")

        # Axis labels
        ax.set_xlabel("X → forward (toward fingertip) [mm]", color="#aaa", fontsize=10)
        ax.set_ylabel("Y → left (toward thumb / object) [mm]", color="#aaa", fontsize=10)

        # Draw axis arrows at origin area
        ax.annotate("", xy=(8, -12), xytext=(0, -12),
                    arrowprops=dict(arrowstyle="-|>", color="#666", lw=1.5))
        ax.text(9, -12, "+X", color="#666", fontsize=9, va="center")
        ax.annotate("", xy=(0, -4), xytext=(0, -12),
                    arrowprops=dict(arrowstyle="-|>", color="#666", lw=1.5))
        ax.text(0, -3, "+Y", color="#666", fontsize=9, ha="center")

        # ---- Compute geometry ----
        theta = np.radians(self.joint_angle_deg)

        # Link direction: finger points forward-left when flexed
        #   cos(40°) ≈ 0.77 (mostly forward)
        #   sin(40°) ≈ 0.64 (partly left/up on screen)
        link_dx = np.cos(theta)  # X component of link direction
        link_dy = np.sin(theta)  # Y component of link direction
        link_dir = np.array([link_dx, link_dy])

        axle = np.array([0.0, 0.0])
        tip = axle + self.link_len * link_dir

        # ============================================================
        # STEP 0: Just the coordinate frame and finger
        # ============================================================

        # Always draw the finger
        # Proximal (fixed to hand, going backward)
        prox_end = axle - 12 * link_dir
        ax.plot([axle[0], prox_end[0]], [axle[1], prox_end[1]],
                color="#555", lw=10, solid_capstyle="round", zorder=1)
        ax.text(prox_end[0] - 2, prox_end[1] - 3, "proximal\n(to palm)",
                color="#666", fontsize=8, ha="center")

        # Distal (the part we're analyzing)
        ax.plot([axle[0], tip[0]], [axle[1], tip[1]],
                color="#a8dadc", lw=10, solid_capstyle="round", zorder=2)

        # Axle (PIP joint)
        ax.add_patch(Circle(axle, 1.8, fc="#e94560", ec="white", lw=2, zorder=6))
        ax.text(-3, 3, "PIP axle\n(magnet here)", color="#e94560", fontsize=8, ha="right")

        # Fingertip
        ax.add_patch(Circle(tip, 1.2, fc="#a8dadc", ec="white", lw=1, zorder=6))
        ax.text(tip[0] + 2, tip[1] + 2, "fingertip", color="#a8dadc", fontsize=9)

        if self.step == 0:
            ax.set_title(
                "Step 0: Geometry — finger flexed 40° in XY plane\n"
                "Axle is in Z (out of page). Finger curls toward +Y (left/thumb side).",
                color="white", fontsize=12, pad=15,
            )
            # Annotate the angle
            arc = Arc(axle, 12, 12, angle=0, theta1=0, theta2=self.joint_angle_deg,
                      color="#ffd166", lw=1.5, ls="--")
            ax.add_patch(arc)
            ax.text(8, 3, f"{self.joint_angle_deg}°", color="#ffd166", fontsize=10)

            # Verify text
            ax.text(0.02, 0.02,
                    f"CHECK: link_dir = ({link_dx:.2f}, {link_dy:.2f})\n"
                    f"  X component = cos(40°) = {link_dx:.2f} → mostly forward ✓\n"
                    f"  Y component = sin(40°) = {link_dy:.2f} → partly left ✓\n"
                    f"Tip position = ({tip[0]:.1f}, {tip[1]:.1f}) mm\n"
                    f"  tip is forward-left of axle ✓",
                    transform=ax.transAxes, color="#44ff44", fontsize=9,
                    family="monospace", va="bottom",
                    bbox=dict(boxstyle="round", fc="#1a1a2e", ec="#44ff44", lw=1))

        # ============================================================
        # STEP 1: Object and grip force direction
        # ============================================================
        if self.step >= 1:
            # Object is on the LEFT (+Y side), held against thumb
            # Draw object as a block above/left of the fingertip
            obj_center = tip + np.array([0, 6])
            ax.add_patch(plt.Rectangle(
                (obj_center[0] - 5, obj_center[1] - 3), 10, 6,
                fc="#334455", ec="#667788", lw=2, zorder=1))
            ax.text(obj_center[0], obj_center[1], "object", color="#99aabb",
                    fontsize=9, ha="center", va="center")

            # Thumb on far side of object
            ax.text(obj_center[0], obj_center[1] + 6, "← thumb on other side",
                    color="#888", fontsize=8, ha="center")

        if self.step == 1:
            ax.set_title(
                "Step 1: Object is on the +Y side (left, toward thumb)\n"
                "The finger pushes the object in +Y. Object pushes back on finger in −Y.",
                color="white", fontsize=12, pad=15,
            )

            # The finger pushes the object in +Y (toward thumb)
            # So the REACTION force on the fingertip from the object is in -Y
            # (object pushes fingertip AWAY from object, back toward right)

            F_on_tip = np.array([0.0, -self.grip_force])  # object pushes finger in -Y
            self.arrow(tip, F_on_tip, "#ff6b6b",
                       f"F_contact = {self.grip_force:.0f}N\n(object pushes finger in −Y)",
                       label_offset=(8, -3), scale=2.0)

            # Show the direction the finger is pushing
            F_finger_pushes = np.array([0.0, self.grip_force])
            self.arrow(tip + np.array([0, 0.5]), F_finger_pushes, "#ffd166",
                       f"finger pushes +Y\n(toward object)",
                       label_offset=(-10, 0), scale=1.5, lw=1.5)

            ax.text(0.02, 0.02,
                    "CHECK: Finger pushes object in +Y (up on screen, toward thumb)\n"
                    "  → Newton's 3rd: object pushes fingertip in −Y (down on screen)\n"
                    "  → F_contact on fingertip = (0, −5) N\n"
                    "  This is correct: object resists the finger, pushes it back ✓",
                    transform=ax.transAxes, color="#44ff44", fontsize=9,
                    family="monospace", va="bottom",
                    bbox=dict(boxstyle="round", fc="#1a1a2e", ec="#44ff44", lw=1))

        # ============================================================
        # STEP 2: Free body diagram — what must the axle provide?
        # ============================================================
        if self.step >= 2:
            F_contact = np.array([0.0, -self.grip_force])

            # The contact force on the tip is (0, -5)N
            # This force acts on the distal phalanx at the tip.
            # The distal phalanx is ONLY connected to the proximal via the axle.
            # (The human finger provides the grip force, but the reaction from
            #  the object goes: object → glove tip → through rigid PLA → axle)
            #
            # Free body of the distal phalanx (the PLA shell):
            #   F_contact at tip + R_axle at axle = 0  (translation)
            #   Moment about axle from F_contact = torque the joint must resist
            #
            # BUT WAIT: The PLA is rigid. Like you said, there is no compression
            # in the PLA. The force goes through the rigid link to the axle as
            # a FORCE, not as deformation of the link itself.
            #
            # Translational equilibrium:
            #   R_axle + F_contact = 0
            #   R_axle = -F_contact = (0, +5) N
            #
            # The axle pushes the distal phalanx in +Y to keep it from sliding in -Y.

            R_axle = -F_contact  # (0, +5)

            # Draw contact force
            self.arrow(tip, F_contact, "#ff6b6b",
                       f"F_contact = (0, {F_contact[1]:.0f})N",
                       label_offset=(8, -3), scale=2.0)

            # Draw axle reaction
            self.arrow(axle, R_axle, "#e94560",
                       f"R_axle = (0, +{R_axle[1]:.0f})N",
                       label_offset=(-8, 3), scale=2.0)

        if self.step == 2:
            ax.set_title(
                "Step 2: Free body of distal phalanx (the rigid PLA shell)\n"
                "Only the axle connects distal to proximal. Axle must balance the contact force.",
                color="white", fontsize=12, pad=15,
            )

            # Also show moment
            M = tip[0] * F_contact[1] - tip[1] * F_contact[0]

            ax.text(0.02, 0.02,
                    f"FREE BODY of distal phalanx (PLA shell):\n"
                    f"  F_contact at tip = (0, {F_contact[1]:.0f}) N\n"
                    f"  R_axle at axle   = (0, +{R_axle[1]:.0f}) N\n"
                    f"  Sum of forces    = (0, 0) ✓  (equilibrium)\n"
                    f"\n"
                    f"  Moment about axle = {M:.1f} N·mm\n"
                    f"  This moment is resisted by your finger muscles\n"
                    f"  (tendon torque through the human PIP joint, NOT through the glove)\n"
                    f"\n"
                    f"  KEY: The PLA is rigid — no compression/deformation.\n"
                    f"  The full {self.grip_force:.0f}N reaction goes through the axle as a FORCE.\n"
                    f"  The axle pushes the shell in +Y (toward the object). ✓",
                    transform=ax.transAxes, color="#44ff44", fontsize=9,
                    family="monospace", va="bottom",
                    bbox=dict(boxstyle="round", fc="#1a1a2e", ec="#44ff44", lw=1))

        # ============================================================
        # STEP 3: But wait — does it actually go through the glove axle?
        # ============================================================
        if self.step == 3:
            F_contact = np.array([0.0, -self.grip_force])
            R_axle = -F_contact

            # Draw forces
            self.arrow(tip, F_contact, "#ff6b6b",
                       f"F_contact (0, {F_contact[1]:.0f})N",
                       label_offset=(8, -3), scale=2.0)
            self.arrow(axle, R_axle, "#e94560",
                       f"R_axle (0, +{R_axle[1]:.0f})N",
                       label_offset=(-8, 3), scale=2.0)

            ax.set_title(
                "Step 3: Does this force ACTUALLY go through the glove's PIP axle?\n"
                "Your concern: the human finger is inside — does it steal the force?",
                color="white", fontsize=12, pad=15,
            )

            ax.text(0.02, 0.02,
                    "THINK ABOUT IT:\n"
                    "\n"
                    "Your finger PUSHES the glove tip into the object (+Y).\n"
                    "The object pushes the glove tip BACK (−Y).\n"
                    "\n"
                    "These are NOT two separate forces on the glove shell!\n"
                    "Your finger creates the grip. The object reacts on the OUTSIDE\n"
                    "of the shell. That reaction force travels through the rigid PLA\n"
                    "back to the axle.\n"
                    "\n"
                    "Where does it go at the axle? Into the PROXIMAL shell,\n"
                    "which is held by the back of your hand.\n"
                    "\n"
                    "The force path is:\n"
                    "  finger muscle → human finger → glove tip → object\n"
                    "  object → glove tip → rigid PLA → GLOVE AXLE → proximal shell → hand\n"
                    "\n"
                    "The glove axle IS in the load path. It sees the full force.\n"
                    "The PLA doesn't deform — the force arrives at the axle as\n"
                    "a rigid-body reaction force, not as material strain.\n"
                    "\n"
                    "★ Your slot probably didn't show VISIBLE motion because even\n"
                    "  at 5N and 50 N/mm stiffness → 100µm displacement.\n"
                    "  That's 0.1mm. Invisible to the eye. But the Hall sensor\n"
                    "  can resolve <1µm of magnet motion.",
                    transform=ax.transAxes, color="#ffd166", fontsize=9,
                    family="monospace", va="bottom",
                    bbox=dict(boxstyle="round", fc="#1a1a2e", ec="#ffd166", lw=1))

        # ============================================================
        # STEP 4: Where to put the slot
        # ============================================================
        if self.step == 4:
            F_contact = np.array([0.0, -self.grip_force])
            R_axle = -F_contact

            self.arrow(tip, F_contact, "#ff6b6b",
                       f"F_contact (0, {F_contact[1]:.0f})N",
                       label_offset=(8, -3), scale=2.0)
            self.arrow(axle, R_axle, "#e94560",
                       f"R_axle (0, +{R_axle[1]:.0f})N",
                       label_offset=(-8, 3), scale=2.0)

            ax.set_title(
                "Step 4: Slot direction — the axle needs to move in +Y to be sensed\n"
                "Slot must allow motion in Y (perpendicular to link, toward/away from object)",
                color="white", fontsize=12, pad=15,
            )

            # Draw zoomed axle with slot
            # The slot should allow Y motion (up-down on screen)
            slot_center = axle + np.array([-12, 0])  # draw slot diagram offset to the left

            # Slot visualization (oval bore)
            slot_w = 3.0   # tight in X (along link-ish)
            slot_h = 5.0   # loose in Y (force direction)

            # Housing
            t_el = np.linspace(0, 2*np.pi, 100)
            ax.plot(slot_center[0] + slot_w*np.cos(t_el),
                    slot_center[1] + slot_h*np.sin(t_el),
                    color="#4a8ab5", lw=2.5)
            ax.text(slot_center[0], slot_center[1] + slot_h + 1.5,
                    "slot in bore\n(oval, loose in Y)",
                    color="#4a8ab5", fontsize=8, ha="center")

            # Pin (nominal)
            ax.add_patch(Circle(slot_center, 1.5, fc="#555", ec="#777", lw=1, ls="--"))

            # Pin (displaced in +Y by axle reaction)
            disp_y = 1.5  # exaggerated for visibility
            ax.add_patch(Circle((slot_center[0], slot_center[1] + disp_y),
                                1.5, fc="#e94560", ec="white", lw=2))
            ax.annotate("", xy=(slot_center[0], slot_center[1] + disp_y),
                        xytext=slot_center,
                        arrowprops=dict(arrowstyle="-|>", color="white", lw=2))
            ax.text(slot_center[0] + 3, slot_center[1] + disp_y,
                    "magnet\ndisplaces\nin +Y", color="white", fontsize=8)

            # Label directions on slot
            ax.annotate("", xy=(slot_center[0], slot_center[1] + slot_h),
                        xytext=(slot_center[0], slot_center[1] - slot_h),
                        arrowprops=dict(arrowstyle="<|-|>", color="#ffd166", lw=1.5))
            ax.text(slot_center[0] - 3, slot_center[1],
                    "Y\n(force\ndir)", color="#ffd166", fontsize=8, ha="center")

            # Compliance numbers
            stiffnesses = [20, 50, 100]
            force = self.grip_force
            grad = 30  # mT/mm, rough estimate for 3mm magnet at 1.5mm
            noise_uT = 0.3

            table = "Slot stiffness → displacement → Hall SNR:\n\n"
            for k in stiffnesses:
                d_mm = force / k
                d_um = d_mm * 1000
                dB = d_mm * grad * 1000  # µT
                snr = dB / noise_uT
                table += f"  k={k:3d} N/mm → {d_um:5.0f} µm → ΔB={dB:6.0f} µT → SNR={snr:5.0f}x\n"

            table += (
                f"\n"
                f"  Noise floor: {noise_uT} µT\n"
                f"  SNR > 3 is detectable, > 10 is comfortable.\n"
                f"\n"
                f"  At k=50 N/mm and {force:.0f}N grip:\n"
                f"    displacement = {force/50*1000:.0f} µm = {force/50:.2f} mm\n"
                f"    INVISIBLE to your eye, but SNR ≈ {force/50*grad*1000/noise_uT:.0f}x for the sensor.\n"
                f"\n"
                f"  ★ RECOMMENDATION:\n"
                f"    • Oval bore: tight in X (rotation axis), 0.3-0.5mm clearance in Y\n"
                f"    • Or: elastomer bushing (silicone tube) around the pin\n"
                f"    • Restoring force from elastomer or flexure\n"
                f"    • Read MLX90393 Bx/By — the Y channel IS your force signal"
            )

            ax.text(0.02, 0.02, table,
                    transform=ax.transAxes, color="#44ff44", fontsize=9,
                    family="monospace", va="bottom",
                    bbox=dict(boxstyle="round", fc="#1a1a2e", ec="#44ff44", lw=1))

        ax.set_xlim(-25, 35)
        ax.set_ylim(-18, 30)

        # Step counter
        ax.text(0.98, 0.98, f"Step {self.step} / {self.max_step}",
                transform=ax.transAxes, color="white", fontsize=12,
                ha="right", va="top",
                bbox=dict(boxstyle="round", fc="#333", ec="white", lw=1))

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


def main():
    app = StepByStep()
    app.show()


if __name__ == "__main__":
    main()
