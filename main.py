import asyncio
import json
import sys
from discord.ext import commands
import Chat

startup = True

try:
    with open('config.json') as f:
        config = json.load(f)
except FileNotFoundError:
    print('CONFIG FILE NOT FOUND! using argv fallback.')

if len(sys.argv) > 1:
    config['token'] = sys.argv[1]
if len(sys.argv) > 2:
    config['prefix'] = sys.argv[2]
if len(sys.argv) > 3:
    config['history_backlog'] = sys.argv[3]
if len(sys.argv) > 4:
    config['train_channel'] = sys.argv[3]

bot = commands.Bot(command_prefix=config['prefix'], self_bot=False)


# gets chat history
async def get_hist(c=config['train_channel'], lim=config['history_backlog']):
    chan = bot.get_channel(c)
    history = []
    hist_itr = chan.history(limit=lim)
    async for m in hist_itr:
        history.insert(0, m.content)
        # print(m.content)
    return history

# get response
@bot.command()
async def chat(ctx, *, message):
    async with ctx.channel.typing():
        print('getting response')
        response = Chat.get_response(message)
    if response:
        await ctx.send(response)
    else:
        await ctx.send('i deadass don\'t know how to respond to that')

# tell you when the bot is ready
@bot.event
async def on_ready():
    global startup
    if startup:
        startup = False
        print('Logged in')
        Chat.build_model(await get_hist())


# main function to start the bot
def main():
    loop = asyncio.get_event_loop()
    loop.create_task(bot.start(config['token']))

    try:
        loop.run_forever()
    finally:
        loop.stop()


if __name__ == '__main__':
    main()
