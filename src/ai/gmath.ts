export class GMath {
    static clamp(num: number, max: number, min?: number) {
        if(num > max) num = max;
        if(min && num < min) num = min;
        return num;
    }
}