/**
 * Animated Space Background for PolyAI
 * Features: Parallax stars, twinkling, and nebula nebula effects.
 */

class SpaceBackground {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext('2d');
        this.stars = [];
        this.nebulae = [];
        this.numStars = 400;
        this.numNebulae = 3;
        this.shootingStars = [];

        this.init();
        this.animate();

        window.addEventListener('resize', () => this.resize());
    }

    init() {
        this.resize();
        this.createStars();
        this.createNebulae();
        this.scheduleShootingStar();
    }

    scheduleShootingStar() {
        const delay = Math.random() * 5000 + 3000;
        setTimeout(() => {
            this.createShootingStar();
            this.scheduleShootingStar();
        }, delay);
    }

    createShootingStar() {
        this.shootingStars.push({
            x: Math.random() * this.canvas.width,
            y: Math.random() * (this.canvas.height / 2),
            length: Math.random() * 80 + 20,
            speed: Math.random() * 10 + 5,
            opacity: 1,
            angle: Math.PI / 4 + (Math.random() - 0.5) * 0.2
        });
    }

    resize() {
        const parent = this.canvas.parentElement;
        this.canvas.width = parent.clientWidth;
        this.canvas.height = parent.clientHeight;
        // Re-create stars/nebulae on large resize if needed, 
        // but for now just let them drift.
    }

    createStars() {
        this.stars = [];
        for (let i = 0; i < this.numStars; i++) {
            this.stars.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                size: Math.random() * 1.5 + 0.5,
                opacity: Math.random(),
                speed: Math.random() * 0.05 + 0.01,
                twinkleSpeed: Math.random() * 0.02 + 0.005,
                twinkleFactor: Math.random() * 0.2
            });
        }
    }

    createNebulae() {
        this.nebulae = [];
        const colors = [
            'rgba(79, 70, 229, 0.05)', // Indigo
            'rgba(236, 72, 153, 0.05)', // Pink
            'rgba(59, 130, 246, 0.05)'  // Blue
        ];

        for (let i = 0; i < this.numNebulae; i++) {
            this.nebulae.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                radius: Math.random() * 300 + 200,
                color: colors[i % colors.length],
                vx: (Math.random() - 0.5) * 0.1,
                vy: (Math.random() - 0.5) * 0.1
            });
        }
    }

    draw() {
        this.ctx.fillStyle = '#05070a';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw Nebulae
        this.nebulae.forEach(n => {
            const gradient = this.ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, n.radius);
            gradient.addColorStop(0, n.color);
            gradient.addColorStop(1, 'transparent');

            this.ctx.fillStyle = gradient;
            this.ctx.beginPath();
            this.ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
            this.ctx.fill();

            // Move nebulae slowly
            n.x += n.vx;
            n.y += n.vy;

            // Boundary check
            if (n.x < -n.radius) n.x = this.canvas.width + n.radius;
            if (n.x > this.canvas.width + n.radius) n.x = -n.radius;
            if (n.y < -n.radius) n.y = this.canvas.height + n.radius;
            if (n.y > this.canvas.height + n.radius) n.y = -n.radius;
        });

        // Draw Stars
        this.stars.forEach(s => {
            s.opacity += s.twinkleSpeed;
            if (s.opacity > 1 || s.opacity < 0.3) {
                s.twinkleSpeed = -s.twinkleSpeed;
            }

            this.ctx.fillStyle = `rgba(255, 255, 255, ${s.opacity})`;
            this.ctx.beginPath();
            this.ctx.arc(s.x, s.y, s.size, 0, Math.PI * 2);
            this.ctx.fill();

            // Parallax movement
            s.y += s.speed;
            if (s.y > this.canvas.height) {
                s.y = 0;
                s.x = Math.random() * this.canvas.width;
            }
        });

        // Draw Shooting Stars
        for (let i = this.shootingStars.length - 1; i >= 0; i--) {
            const s = this.shootingStars[i];

            this.ctx.strokeStyle = `rgba(255, 255, 255, ${s.opacity})`;
            this.ctx.lineWidth = 2;
            this.ctx.beginPath();
            this.ctx.moveTo(s.x, s.y);
            this.ctx.lineTo(s.x + Math.cos(s.angle) * s.length, s.y + Math.sin(s.angle) * s.length);
            this.ctx.stroke();

            s.x += Math.cos(s.angle) * s.speed;
            s.y += Math.sin(s.angle) * s.speed;
            s.opacity -= 0.01;

            if (s.opacity <= 0) {
                this.shootingStars.splice(i, 1);
            }
        }
    }

    animate() {
        this.draw();
        requestAnimationFrame(() => this.animate());
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new SpaceBackground('space-bg');
});
