import re
dy = [-2,0,2,0]
dx = [0,2,0,-2]
def solution(places):
    answer = 0
    person_list = []
    for tc in places:
        for y in range(5):
            for x in range(5):
                if tc[y][x] == 'P':
                    person_list.append((y,x))
        print(person_list)

        # search 1 가로 세로
        for y, x in person_list:
            y_check = y
            x_check = x
            for i in range(4):
                y_check +=dy[i]
                x_check +=dx[i]
                if y_check <0 or x_check <0:
                    pass
                elif tc[y_check][x_check] == 'P':


    return answer

places=[["POOOP", "OXXOX", "OPXPX", "OOXOX", "POXXP"], ["POOPX", "OXPXP", "PXXXO", "OXXXO", "OOOPP"], ["PXOPX", "OXOXP", "OXPXX", "OXXXP", "POOXX"], ["OOOXX", "XOOOX", "OOOXX", "OXOOX", "OOOOO"], ["PXPXP", "XPXPX", "PXPXP", "XPXPX", "PXPXP"]]
solution(places)